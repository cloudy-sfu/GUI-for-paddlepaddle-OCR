set VENV=.\venv
copy %VENV%\Scripts\pip_autoremove.py %VENV%\Lib\site-packages\
copy patch\pass_desc_pb2.py %VENV%\Lib\site-packages\paddle\fluid\proto\pass_desc_pb2.py
copy patch\image.py %VENV%\Lib\site-packages\paddle\dataset\image.py
