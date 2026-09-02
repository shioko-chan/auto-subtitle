{
  description = "CUDA development shell for the subtitle pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp";
      flake = false;
    };
  };

  outputs =
    { nixpkgs, llama-cpp-src, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "7.5" ];
        };
      };
      cuda = pkgs.cudaPackages;
      llamaCpp = pkgs.llama-cpp.overrideAttrs (_: {
        version = "0";
        src = llama-cpp-src;
      });
      assCudaRender = pkgs.stdenv.mkDerivation {
        pname = "ass-cuda-render";
        version = "0.1.0";
        src = ./native;
        nativeBuildInputs = [ pkgs.pkg-config cuda.cuda_nvcc ];
        buildInputs = [ pkgs.ffmpeg.dev pkgs.libass.dev cuda.cuda_cudart ];
        buildPhase = ''
          nvcc -std=c++17 -O3 -lineinfo \
            $(pkg-config --cflags libavformat libavcodec libavutil libass) \
            ass_cuda_render.cu -o ass-cuda-render \
            $(pkg-config --libs libavformat libavcodec libavutil libass) \
            -lcudart
        '';
        installPhase = ''
          mkdir -p $out/bin
          install -m755 ass-cuda-render $out/bin/
        '';
      };
      runtimeLibraries = [
        cuda.cuda_cudart
        cuda.libcublas
        cuda.libcusparse
        cuda.libnvjitlink
        pkgs.ffmpeg
        pkgs.espeak-ng
        pkgs.glib
        pkgs.libglvnd
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python311
          pkgs.uv
          pkgs.ffmpeg
          pkgs.espeak-ng
          pkgs.chromium
          pkgs.noto-fonts-monochrome-emoji
          cuda.cuda_nvcc
          cuda.cuda_cccl
          assCudaRender
          pkgs.pkg-config
          pkgs.ffmpeg.dev
          pkgs.libass.dev
          llamaCpp
        ];

        CUDA_HOME = "${cuda.cuda_nvcc}";
        CUDA_PATH = "${cuda.cuda_nvcc}";
        UV_PYTHON = "${pkgs.python311}/bin/python";
        TORCH_CUDA_ARCH_LIST = "7.5";
        # Triton otherwise calls the non-existent NixOS path /sbin/ldconfig.
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
        LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath runtimeLibraries}";

        shellHook = ''
          project_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          if [ -d "$project_root/.venv/bin" ]; then
            export PATH="$project_root/.venv/bin:$PATH"
          fi
          echo "subtitle-pipeline CUDA shell: $(python --version 2>&1), CUDA $(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
        '';
      };
      packages.${system}.ass-cuda-render = assCudaRender;
    };
}
