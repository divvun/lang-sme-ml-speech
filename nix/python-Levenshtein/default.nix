{ lib
, buildPythonPackage
, fetchPypi
, isPy27
, isPy3k
}:

buildPythonPackage rec {
  pname = "python-Levenshtein";
  version = "0.12.2";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256:1xj60gymwx1jl2ra9razx2wk8nb9cv1i7l8d14qsp8a8s7xra8yw";
  };

#   # AttributeError: 'KeywordMapping' object has no attribute 'get'
#   doCheck = !isPy27;

  meta = with lib; {
    description = "python-Levenshtein";
    homepage = "https://github.com/ztane/python-Levenshtein";
    license = licenses.gpl2Plus;
    maintainers = with maintainers; [ ];
  };
}
