{ lib
, buildPythonPackage
, fetchPypi
, isPy27
, isPy3k
, python-Levenshtein
}:

buildPythonPackage rec {
  pname = "jiwer";
  version = "2.2.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256:099mrjm757k5caacl7pvzngaxrsmbl9g9sjg8867dnkwj9i4vmym";
  };

#   # AttributeError: 'KeywordMapping' object has no attribute 'get'
#   doCheck = !isPy27;
  propagatedBuildInputs = [ python-Levenshtein ];
  pythonImportsCheck = [ "jiwer" ];

  meta = with lib; {
    description = "jiwer";
    homepage = "https://github.com/jitsi/jiwer";
    license = licenses.asl20;
    maintainers = with maintainers; [ ];
  };
}
