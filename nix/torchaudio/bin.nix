{ lib
, stdenv
, buildPythonPackage
, fetchurl
, isPy37
, isPy38
, isPy39
, patchelf
, pillow
, python
, pytorch-bin
}:

let
  pyVerNoDot = builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion;
  srcs = import ./binary-hashes.nix version;
  unsupported = throw "Unsupported system";
  version = "0.9.1";
in buildPythonPackage {
  inherit version;

  pname = "torchaudio";

  format = "wheel";

  src = fetchurl srcs."${stdenv.system}-${pyVerNoDot}" or unsupported;

  disabled = !(isPy37 || isPy38 || isPy39);

  nativeBuildInputs = [
    patchelf
  ];

  propagatedBuildInputs = [
    pillow
    pytorch-bin
  ];

  pythonImportsCheck = [ "torchaudio" ];

  postFixup = let
    rpath = lib.makeLibraryPath [ stdenv.cc.cc.lib ];
  in ''
    # Note: after patchelf'ing, libcudart can still not be found. However, this should
    #       not be an issue, because PyTorch is loaded before torchaudio and brings
    #       in the necessary symbols.
    patchelf --set-rpath "${rpath}:${pytorch-bin}/${python.sitePackages}/torch/lib:" \
      "$out/${python.sitePackages}/torchaudio/_torchaudio.so"
  '';

  meta = with lib; {
    description = "PyTorch audio library";
    homepage = "https://pytorch.org/";
    changelog = "https://github.com/pytorch/audio/releases/tag/v${version}";
    license = licenses.bsd3;
    platforms = platforms.linux;
    # maintainers = with maintainers; [ junjihashimoto ];
  };
}