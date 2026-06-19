with open("python/tests/test_python_api.py", "r") as f:
    content = f.read()

content = content.replace("mock_get_kwargs = mock.patch('luna_rust._kwargs.get_julia').start()", "pass")
content = content.replace("mock_get_output = mock.patch('luna_rust.output.get_julia').start()", "pass")
content = content.replace("with mock.patch('luna_rust.output.LunaOutput', autospec=True) as mock_out:", "with mock.patch('luna_rust.LunaOutput', autospec=True) as mock_out:")

with open("python/tests/test_python_api.py", "w") as f:
    f.write(content)
