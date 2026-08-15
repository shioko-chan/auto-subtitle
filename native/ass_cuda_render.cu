#include <cuda_runtime.h>

extern "C" {
#include <ass/ass.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
}

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static AVPixelFormat get_cuda_format(AVCodecContext *, const AVPixelFormat *formats) {
    for (const AVPixelFormat *format = formats; *format != AV_PIX_FMT_NONE; ++format)
        if (*format == AV_PIX_FMT_CUDA)
            return *format;
    return AV_PIX_FMT_NONE;
}

static void fail(const std::string &message) {
    std::fprintf(stderr, "ass-cuda-render: %s\n", message.c_str());
    std::exit(1);
}

static void check_av(int result, const char *operation) {
    if (result >= 0)
        return;
    char detail[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(result, detail, sizeof(detail));
    fail(std::string(operation) + ": " + detail);
}

static void check_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess)
        fail(std::string(operation) + ": " + cudaGetErrorString(result));
}

__global__ static void blend_luma(
    uint8_t *plane, int pitch, int frame_width, int frame_height,
    const uint8_t *mask, int mask_stride, int width, int height,
    int dst_x, int dst_y, float opacity, float subtitle_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int frame_x = dst_x + x;
    int frame_y = dst_y + y;
    if (frame_x < 0 || frame_y < 0 || frame_x >= frame_width || frame_y >= frame_height)
        return;
    float alpha = (mask[y * mask_stride + x] / 255.0f) * opacity;
    uint8_t *pixel = plane + frame_y * pitch + frame_x;
    *pixel = static_cast<uint8_t>(lrintf(*pixel * (1.0f - alpha) + subtitle_y * alpha));
}

__global__ static void blend_chroma(
    uint8_t *plane, int pitch, int frame_width, int frame_height,
    const uint8_t *mask, int mask_stride, int width, int height,
    int dst_x, int dst_y, float opacity, float subtitle_u, float subtitle_v
) {
    int chroma_x = blockIdx.x * blockDim.x + threadIdx.x;
    int chroma_y = blockIdx.y * blockDim.y + threadIdx.y;
    int first_x = dst_x / 2;
    int first_y = dst_y / 2;
    int global_cx = first_x + chroma_x;
    int global_cy = first_y + chroma_y;
    if (global_cx < 0 || global_cy < 0 || global_cx >= (frame_width + 1) / 2 ||
        global_cy >= (frame_height + 1) / 2)
        return;

    float coverage = 0.0f;
    for (int oy = 0; oy < 2; ++oy) {
        for (int ox = 0; ox < 2; ++ox) {
            int local_x = global_cx * 2 + ox - dst_x;
            int local_y = global_cy * 2 + oy - dst_y;
            if (local_x >= 0 && local_y >= 0 && local_x < width && local_y < height)
                coverage += mask[local_y * mask_stride + local_x] / 255.0f;
        }
    }
    float alpha = coverage * 0.25f * opacity;
    uint8_t *pixel = plane + global_cy * pitch + global_cx * 2;
    pixel[0] = static_cast<uint8_t>(lrintf(pixel[0] * (1.0f - alpha) + subtitle_u * alpha));
    pixel[1] = static_cast<uint8_t>(lrintf(pixel[1] * (1.0f - alpha) + subtitle_v * alpha));
}

struct Renderer {
    ASS_Library *library = nullptr;
    ASS_Renderer *renderer = nullptr;
    ASS_Track *track = nullptr;
    struct DeviceImage {
        int width, height, stride, dst_x, dst_y;
        float opacity, y, u, v;
        uint8_t *mask;
    };
    std::vector<DeviceImage> images;

    ~Renderer() {
        for (const DeviceImage &image : images)
            cudaFree(image.mask);
        if (track)
            ass_free_track(track);
        if (renderer)
            ass_renderer_done(renderer);
        if (library)
            ass_library_done(library);
    }
};

static void rgb_to_yuv(
    int red, int green, int blue, bool bt709, bool full_range,
    float &y, float &u, float &v
) {
    if (bt709 && full_range) {
        y = 0.212600f * red + 0.715200f * green + 0.072200f * blue;
        u = 128.0f - 0.114572f * red - 0.385428f * green + 0.500000f * blue;
        v = 128.0f + 0.500000f * red - 0.454153f * green - 0.045847f * blue;
    } else if (bt709) {
        y = 16.0f + 0.182586f * red + 0.614231f * green + 0.062007f * blue;
        u = 128.0f - 0.100644f * red - 0.338572f * green + 0.439216f * blue;
        v = 128.0f + 0.439216f * red - 0.398942f * green - 0.040274f * blue;
    } else if (full_range) {
        y = 0.299000f * red + 0.587000f * green + 0.114000f * blue;
        u = 128.0f - 0.168736f * red - 0.331264f * green + 0.500000f * blue;
        v = 128.0f + 0.500000f * red - 0.418688f * green - 0.081312f * blue;
    } else {
        y = 16.0f + 0.256788f * red + 0.504129f * green + 0.097906f * blue;
        u = 128.0f - 0.148223f * red - 0.290993f * green + 0.439216f * blue;
        v = 128.0f + 0.439216f * red - 0.367788f * green - 0.071427f * blue;
    }
}

static void render_ass(Renderer &state, AVFrame *frame, int64_t time_ms) {
    int changed = 0;
    ASS_Image *images = ass_render_frame(state.renderer, state.track, time_ms, &changed);
    bool bt709 = frame->colorspace == AVCOL_SPC_BT709 ||
                 (frame->colorspace == AVCOL_SPC_UNSPECIFIED && frame->height >= 720);
    bool full_range = frame->color_range == AVCOL_RANGE_JPEG;
    if (changed || (state.images.empty() && images)) {
        for (const Renderer::DeviceImage &image : state.images)
            check_cuda(cudaFree(image.mask), "free cached subtitle mask");
        state.images.clear();
        for (ASS_Image *image = images; image; image = image->next) {
            if (image->w <= 0 || image->h <= 0)
                continue;
            Renderer::DeviceImage cached = {};
            cached.width = image->w;
            cached.height = image->h;
            cached.stride = image->stride;
            cached.dst_x = image->dst_x;
            cached.dst_y = image->dst_y;
            int red = (image->color >> 24) & 0xff;
            int green = (image->color >> 16) & 0xff;
            int blue = (image->color >> 8) & 0xff;
            cached.opacity = (255 - (image->color & 0xff)) / 255.0f;
            rgb_to_yuv(
                red, green, blue, bt709, full_range,
                cached.y, cached.u, cached.v
            );
            size_t required =
                static_cast<size_t>(image->stride) * (image->h - 1) + image->w;
            check_cuda(cudaMalloc(&cached.mask, required), "allocate subtitle mask");
            check_cuda(
                cudaMemcpy2D(
                    cached.mask, image->stride, image->bitmap, image->stride,
                    image->w, image->h, cudaMemcpyHostToDevice
                ),
                "upload subtitle mask"
            );
            state.images.push_back(cached);
        }
    }

    dim3 threads(16, 16);
    for (const Renderer::DeviceImage &image : state.images) {
        dim3 luma_blocks((image.width + 15) / 16, (image.height + 15) / 16);
        blend_luma<<<luma_blocks, threads>>>(
            frame->data[0], frame->linesize[0], frame->width, frame->height,
            image.mask, image.stride, image.width, image.height,
            image.dst_x, image.dst_y, image.opacity, image.y
        );
        int chroma_width = (image.width + 2) / 2 + 1;
        int chroma_height = (image.height + 2) / 2 + 1;
        dim3 chroma_blocks((chroma_width + 15) / 16, (chroma_height + 15) / 16);
        blend_chroma<<<chroma_blocks, threads>>>(
            frame->data[1], frame->linesize[1], frame->width, frame->height,
            image.mask, image.stride, image.width, image.height,
            image.dst_x, image.dst_y, image.opacity, image.u, image.v
        );
        check_cuda(cudaGetLastError(), "launch subtitle blend");
    }
    check_cuda(cudaDeviceSynchronize(), "finish subtitle blend");
}

struct Output {
    AVFormatContext *format = nullptr;
    AVCodecContext *encoder = nullptr;
    AVStream *stream = nullptr;
    bool header_written = false;

    ~Output() {
        if (encoder)
            avcodec_free_context(&encoder);
        if (format) {
            if (!(format->oformat->flags & AVFMT_NOFILE) && format->pb)
                avio_closep(&format->pb);
            avformat_free_context(format);
        }
    }
};

static void initialize_output(
    Output &output, const char *path, AVCodecContext *decoder,
    AVStream *input_stream, const AVFrame *first_frame,
    const std::string &preset, int cq
) {
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec)
        fail("h264_nvenc encoder is unavailable");
    check_av(avformat_alloc_output_context2(&output.format, nullptr, "mp4", path),
             "create output container");
    output.encoder = avcodec_alloc_context3(codec);
    if (!output.encoder)
        fail("allocate NVENC context");
    output.encoder->width = first_frame->width;
    output.encoder->height = first_frame->height;
    output.encoder->pix_fmt = AV_PIX_FMT_CUDA;
    output.encoder->time_base = input_stream->time_base;
    output.encoder->framerate = av_guess_frame_rate(nullptr, input_stream, nullptr);
    output.encoder->sample_aspect_ratio = first_frame->sample_aspect_ratio;
    output.encoder->color_range = input_stream->codecpar->color_range != AVCOL_RANGE_UNSPECIFIED
        ? input_stream->codecpar->color_range
        : decoder->color_range;
    output.encoder->colorspace = input_stream->codecpar->color_space != AVCOL_SPC_UNSPECIFIED
        ? input_stream->codecpar->color_space
        : decoder->colorspace;
    output.encoder->color_primaries =
        input_stream->codecpar->color_primaries != AVCOL_PRI_UNSPECIFIED
        ? input_stream->codecpar->color_primaries
        : decoder->color_primaries;
    output.encoder->color_trc =
        input_stream->codecpar->color_trc != AVCOL_TRC_UNSPECIFIED
        ? input_stream->codecpar->color_trc
        : decoder->color_trc;
    double frames_per_second = av_q2d(output.encoder->framerate);
    if (!std::isfinite(frames_per_second) || frames_per_second <= 0)
        frames_per_second = 30.0;
    int64_t target_bit_rate = static_cast<int64_t>(
        first_frame->width * first_frame->height * frames_per_second * 0.040
    );
    target_bit_rate = std::max<int64_t>(2'000'000, target_bit_rate);
    if (decoder->bit_rate > 0)
        target_bit_rate = std::min(target_bit_rate, decoder->bit_rate);
    output.encoder->bit_rate = target_bit_rate;
    output.encoder->rc_max_rate = target_bit_rate * 5 / 4;
    output.encoder->rc_buffer_size = target_bit_rate * 2;
    output.encoder->hw_frames_ctx = av_buffer_ref(first_frame->hw_frames_ctx);
    if (!output.encoder->hw_frames_ctx)
        fail("decoder did not expose CUDA frame context");
    if (output.format->oformat->flags & AVFMT_GLOBALHEADER)
        output.encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    av_opt_set(output.encoder->priv_data, "preset", preset.c_str(), 0);
    av_opt_set(output.encoder->priv_data, "tune", "hq", 0);
    av_opt_set(output.encoder->priv_data, "rc", "vbr", 0);
    av_opt_set_int(output.encoder->priv_data, "cq", cq, 0);
    av_opt_set(output.encoder->priv_data, "b_ref_mode", "middle", 0);
    av_opt_set_int(output.encoder->priv_data, "spatial-aq", 1, 0);
    check_av(avcodec_open2(output.encoder, codec, nullptr), "open h264_nvenc");
    output.stream = avformat_new_stream(output.format, nullptr);
    if (!output.stream)
        fail("create output video stream");
    output.stream->time_base = output.encoder->time_base;
    check_av(avcodec_parameters_from_context(output.stream->codecpar, output.encoder),
             "copy encoder parameters");
    if (!(output.format->oformat->flags & AVFMT_NOFILE))
        check_av(avio_open(&output.format->pb, path, AVIO_FLAG_WRITE), "open output file");
    check_av(avformat_write_header(output.format, nullptr), "write output header");
    output.header_written = true;
}

static void drain_encoder(Output &output, AVFrame *frame) {
    check_av(avcodec_send_frame(output.encoder, frame), "send frame to NVENC");
    AVPacket *packet = av_packet_alloc();
    if (!packet)
        fail("allocate encoded packet");
    while (true) {
        int result = avcodec_receive_packet(output.encoder, packet);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
            break;
        check_av(result, "receive NVENC packet");
        av_packet_rescale_ts(packet, output.encoder->time_base, output.stream->time_base);
        packet->stream_index = output.stream->index;
        check_av(av_interleaved_write_frame(output.format, packet), "write encoded packet");
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
}

static AVFrame *copy_cuda_frame(const AVFrame *source) {
    if (!source->hw_frames_ctx)
        fail("decoded frame did not expose CUDA frame context");
    AVHWFramesContext *frames = reinterpret_cast<AVHWFramesContext *>(source->hw_frames_ctx->data);
    if (frames->sw_format != AV_PIX_FMT_NV12)
        fail("only 8-bit NV12 CUDA frames are currently supported");
    AVFrame *destination = av_frame_alloc();
    if (!destination)
        fail("allocate CUDA output frame");
    destination->format = AV_PIX_FMT_CUDA;
    destination->width = source->width;
    destination->height = source->height;
    destination->hw_frames_ctx = av_buffer_ref(source->hw_frames_ctx);
    check_av(av_hwframe_get_buffer(destination->hw_frames_ctx, destination, 0),
             "allocate CUDA output surface");
    check_cuda(
        cudaMemcpy2D(
            destination->data[0], destination->linesize[0],
            source->data[0], source->linesize[0], source->width, source->height,
            cudaMemcpyDeviceToDevice
        ),
        "copy decoded luma"
    );
    check_cuda(
        cudaMemcpy2D(
            destination->data[1], destination->linesize[1],
            source->data[1], source->linesize[1], source->width, (source->height + 1) / 2,
            cudaMemcpyDeviceToDevice
        ),
        "copy decoded chroma"
    );
    check_av(av_frame_copy_props(destination, source), "copy frame properties");
    return destination;
}

int main(int argc, char **argv) {
    std::string input, ass_path, output_path, preset = "p4";
    int cq = 23;
    for (int index = 1; index < argc; ++index) {
        if (!std::strcmp(argv[index], "--input") && index + 1 < argc)
            input = argv[++index];
        else if (!std::strcmp(argv[index], "--ass") && index + 1 < argc)
            ass_path = argv[++index];
        else if (!std::strcmp(argv[index], "--output") && index + 1 < argc)
            output_path = argv[++index];
        else if (!std::strcmp(argv[index], "--preset") && index + 1 < argc)
            preset = argv[++index];
        else if (!std::strcmp(argv[index], "--cq") && index + 1 < argc)
            cq = std::atoi(argv[++index]);
        else
            fail("usage: ass-cuda-render --input VIDEO --ass FILE --output MP4 [--preset p4] [--cq 23]");
    }
    if (input.empty() || ass_path.empty() || output_path.empty())
        fail("input, ASS subtitle, and output are required");

    AVFormatContext *input_format = nullptr;
    check_av(avformat_open_input(&input_format, input.c_str(), nullptr, nullptr), "open input");
    check_av(avformat_find_stream_info(input_format, nullptr), "read input streams");
    int video_index = av_find_best_stream(input_format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    check_av(video_index, "find video stream");
    AVStream *video_stream = input_format->streams[video_index];
    const AVPacketSideData *matrix_data = av_packet_side_data_get(
        video_stream->codecpar->coded_side_data,
        video_stream->codecpar->nb_coded_side_data,
        AV_PKT_DATA_DISPLAYMATRIX
    );
    const uint8_t *matrix = matrix_data ? matrix_data->data : nullptr;
    if (matrix && std::fabs(av_display_rotation_get(reinterpret_cast<const int32_t *>(matrix))) > 0.01)
        fail("rotated video is not supported by the CUDA renderer");

    const char *decoder_name = nullptr;
    switch (video_stream->codecpar->codec_id) {
        case AV_CODEC_ID_AV1: decoder_name = "av1_cuvid"; break;
        case AV_CODEC_ID_H264: decoder_name = "h264_cuvid"; break;
        case AV_CODEC_ID_HEVC: decoder_name = "hevc_cuvid"; break;
        case AV_CODEC_ID_VP9: decoder_name = "vp9_cuvid"; break;
        default: break;
    }
    const AVCodec *decoder_codec = decoder_name
        ? avcodec_find_decoder_by_name(decoder_name)
        : nullptr;
    if (!decoder_codec)
        fail("video decoder is unavailable");
    AVCodecContext *decoder = avcodec_alloc_context3(decoder_codec);
    if (!decoder)
        fail("allocate decoder");
    check_av(avcodec_parameters_to_context(decoder, video_stream->codecpar), "copy decoder parameters");
    decoder->pkt_timebase = video_stream->time_base;
    decoder->get_format = get_cuda_format;
    AVDictionary *device_options = nullptr;
    av_dict_set(&device_options, "primary_ctx", "1", 0);
    AVBufferRef *device = nullptr;
    check_av(av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, "0", device_options, 0),
             "create CUDA decode device");
    av_dict_free(&device_options);
    decoder->hw_device_ctx = av_buffer_ref(device);
    check_av(avcodec_open2(decoder, decoder_codec, nullptr), "open CUDA decoder");

    Renderer subtitles;
    subtitles.library = ass_library_init();
    subtitles.renderer = subtitles.library ? ass_renderer_init(subtitles.library) : nullptr;
    if (!subtitles.renderer)
        fail("initialize libass");
    ass_set_frame_size(subtitles.renderer, decoder->width, decoder->height);
    ass_set_storage_size(subtitles.renderer, decoder->width, decoder->height);
    ass_set_fonts(subtitles.renderer, nullptr, nullptr, 1, nullptr, 1);
    subtitles.track = ass_read_file(subtitles.library, const_cast<char *>(ass_path.c_str()), nullptr);
    if (!subtitles.track)
        fail("read ASS subtitle");

    Output output;
    AVPacket *packet = av_packet_alloc();
    AVFrame *decoded = av_frame_alloc();
    if (!packet || !decoded)
        fail("allocate decode objects");
    AVRational frame_rate = av_guess_frame_rate(input_format, video_stream, nullptr);
    int64_t frame_step = 1;
    if (frame_rate.num > 0 && frame_rate.den > 0) {
        frame_step = std::max<int64_t>(
            1, av_rescale_q(1, av_inv_q(frame_rate), video_stream->time_base)
        );
    }
    int64_t last_encoder_pts = AV_NOPTS_VALUE;
    int64_t corrected_timestamps = 0;
    auto process_frames = [&]() {
        while (true) {
            int result = avcodec_receive_frame(decoder, decoded);
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
                break;
            check_av(result, "receive decoded frame");
            if (decoded->format != AV_PIX_FMT_CUDA || !decoded->hw_frames_ctx)
                fail(
                    "decoder returned a non-CUDA frame (format=" +
                    std::to_string(decoded->format) + ")"
                );
            if (!output.header_written)
                initialize_output(output, output_path.c_str(), decoder, video_stream, decoded, preset, cq);
            AVFrame *rendered = copy_cuda_frame(decoded);
            int64_t source_timestamp = rendered->best_effort_timestamp;
            if (source_timestamp == AV_NOPTS_VALUE)
                source_timestamp = rendered->pts;
            if (source_timestamp == AV_NOPTS_VALUE)
                source_timestamp = last_encoder_pts == AV_NOPTS_VALUE
                    ? 0
                    : last_encoder_pts + frame_step;
            int64_t time_ms = av_rescale_q(
                source_timestamp, video_stream->time_base, AVRational{1, 1000}
            );
            render_ass(subtitles, rendered, time_ms);
            int64_t encoder_pts = source_timestamp;
            if (last_encoder_pts != AV_NOPTS_VALUE && encoder_pts <= last_encoder_pts) {
                encoder_pts = last_encoder_pts + frame_step;
                ++corrected_timestamps;
            }
            rendered->pts = encoder_pts;
            last_encoder_pts = encoder_pts;
            drain_encoder(output, rendered);
            av_frame_free(&rendered);
            av_frame_unref(decoded);
        }
    };
    while (av_read_frame(input_format, packet) >= 0) {
        if (packet->stream_index == video_index) {
            check_av(avcodec_send_packet(decoder, packet), "send packet to decoder");
            process_frames();
        }
        av_packet_unref(packet);
    }
    check_av(avcodec_send_packet(decoder, nullptr), "flush decoder");
    process_frames();
    if (!output.header_written)
        fail("input produced no video frames");
    drain_encoder(output, nullptr);
    check_av(av_write_trailer(output.format), "write output trailer");
    std::fprintf(
        stderr,
        "ass-cuda-render: corrected %lld non-monotonic frame timestamps\n",
        static_cast<long long>(corrected_timestamps)
    );

    av_frame_free(&decoded);
    av_packet_free(&packet);
    av_buffer_unref(&device);
    avcodec_free_context(&decoder);
    avformat_close_input(&input_format);
    return 0;
}
