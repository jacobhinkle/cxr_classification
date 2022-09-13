{
  description = "PyTorch code for deep learning on chest X-rays (CXR)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-22.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # See https://old.reddit.com/r/Python/comments/npu66t/reproducible_python_environment_with_nix_flake/
  outputs = { self, nixpkgs, flake-utils }: 
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        python = "python310";
        pkgs = (import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [
          ];
        });
        # core pkgs are those required to run headless scripts
        corePythonPkgs = ps: with ps; [
          numpy
          pandas
          pytorchWithCuda
          scikit-learn
          scipy
          torchvision
          tqdm
        ];
        corePythonEnv = pkgs.${python}.withPackages corePythonPkgs;
        jupyterPythonEnv = pkgs.${python}.withPackages (ps: with ps;
          ((corePythonPkgs ps) ++ [
            ipympl  # for %matplotlib widget
            jupyterlab
            matplotlib
            # not strictly for jupyter but still useful for development
            black
          ]));
      in rec {
        packages = {
          download = pkgs.stdenv.mkDerivation {
            name = "download";
            propagatedBuildInputs = with pkgs; [
              google-cloud-sdk
              ./download_mimic_cxr_jpg.sh
            ];
          };
        };
        apps = rec {
          default = jupyter;
          jupyter = {
            type = "app";
            # Note that this is not a full command line; do not include
            # arguments. If you would like to provide a command line, see the
            # "foo" example further down in this file
            program = "${jupyterPythonEnv}/bin/jupyter";
          };
          download = with pkgs; {
            type = "app";
            program = "${packages.download}/download_mimic_cxr_jpg.sh";
          };
        };
        devShell = pkgs.mkShell { buildInputs = with pkgs; [ jupyterPythonEnv ]; };
      });
}
