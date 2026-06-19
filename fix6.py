with open("python/tests/test_python_api.py", "r") as f:
    content = f.read()

content = content.replace("with mock.patch('luna_rust.LunaOutput')", "with mock.patch('luna_rust.LunaOutput', autospec=True)")

with open("python/tests/test_python_api.py", "w") as f:
    f.write(content)
