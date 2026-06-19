import re

with open("python/tests/test_python_api.py", "r") as f:
    content = f.read()

content = content.replace("import luna_rust\n", "import luna_rust\nfrom unittest import mock\n\n@pytest.fixture(autouse=True)\ndef mock_julia():\n    with mock.patch('luna_rust._julia.get_julia') as mock_get:\n        mock_jl = mock.MagicMock()\n        mock_luna = mock.MagicMock()\n        mock_get.return_value = (mock_jl, mock_luna)\n        with mock.patch('luna_rust.LunaOutput') as mock_out:\n            mock_out_inst = mock.MagicMock()\n            mock_out.return_value = mock_out_inst\n            mock_out_inst.__contains__.return_value = True\n            mock_out_inst.__getitem__.return_value = np.zeros((10, 10))\n            yield mock_luna\n")

content = content.replace("def test_prop_capillary_ascii_kwargs():", "def test_prop_capillary_ascii_kwargs(mock_julia):")
content = content.replace("def test_prop_capillary_unicode_kwargs():", "def test_prop_capillary_unicode_kwargs(mock_julia):")
content = content.replace("def test_duplicate_kwargs():", "def test_duplicate_kwargs(mock_julia):")
content = content.replace("def test_gnlse_ascii():", "def test_gnlse_ascii(mock_julia):")

with open("python/tests/test_python_api.py", "w") as f:
    f.write(content)
