{
  description = "CUDA development shell for the subtitle pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      cuda = pkgs.cudaPackages;
      runtimeLibraries = [
        cuda.cuda_cudart
        cuda.libcublas
        cuda.libcusparse
        pkgs.ffmpeg
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
          cuda.cuda_nvcc
          cuda.cuda_cccl
        ];

        CUDA_HOME = "${cuda.cuda_nvcc}";
        CUDA_PATH = "${cuda.cuda_nvcc}";
        UV_PYTHON = "${pkgs.python311}/bin/python";
        TORCH_CUDA_ARCH_LIST = "7.5";
        # Triton otherwise calls the non-existent NixOS path /sbin/ldconfig.
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
        LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath runtimeLibraries}";

        shellHook = ''
          echo "subtitle-pipeline CUDA shell: $(python --version 2>&1), CUDA $(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
        '';
      };
    };
}
