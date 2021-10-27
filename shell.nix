let pkgs = import <nixpkgs> {};
    torchaudio-bin = ps: ps.callPackage ./nix/torchaudio/bin.nix {};
    python-Levenshtein = ps: ps.callPackage ./nix/python-Levenshtein/default.nix {};
    jiwer = ps: ps.callPackage ./nix/jiwer/default.nix {};
    python38 = pkgs.python38.withPackages(ps: [ 
      ps.pytorch-bin
      (torchaudio-bin ps)
      ps.numpy
      ps.librosa
      ps.unidecode
      ps.matplotlib
      ps.scipy
      ps.tensorflow-bin
      ps.transformers
      ps.datasets
      (jiwer ps)
      ps.pillow
    ]);
in pkgs.mkShell {
    nativeBuildInputs = [
      pkgs.cudatoolkit
    ];
    buildInputs = [
      pkgs.linuxPackages.nvidia_x11
      pkgs.cudnn
      python38
    ];
    
  shellHook = ''
    export PATH=$PATH:/run/current-system/sw/bin
  '';
}