with open("python/tests/test_python_api.py", "r") as f:
    content = f.read()

content = content.replace("with mock.patch('luna_rust._julia.get_julia') as mock_get:", "with mock.patch('luna_rust.get_julia') as mock_get:\n        mock_get_kwargs = mock.patch('luna_rust._kwargs.get_julia').start()\n        mock_get_kwargs.return_value = (mock_jl, mock_luna)\n        mock_get_output = mock.patch('luna_rust.output.get_julia').start()\n        mock_get_output.return_value = (mock_jl, mock_luna)\n")

with open("python/tests/test_python_api.py", "w") as f:
    f.write(content)
