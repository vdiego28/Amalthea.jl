with open("python/tests/test_python_api.py", "r") as f:
    content = f.read()

content = content.replace("with mock.patch('luna_rust.get_julia') as mock_get:", "with mock.patch('luna_rust._julia.get_julia') as mock_get:")
content = content.replace("with mock.patch('luna_rust.LunaOutput', autospec=True) as mock_out:", "with mock.patch('luna_rust.output.LunaOutput', autospec=True) as mock_out:")

with open("python/tests/test_python_api.py", "w") as f:
    f.write(content)
