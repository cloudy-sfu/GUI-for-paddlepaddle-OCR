# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.pyw'],
             pathex=[],
             binaries=[('venv/Lib/site-packages/paddle/libs', 'paddle/libs'),
                       ('geos_c.dll', '_pyinstaller_hooks_contrib/hooks/stdhooks/geos_c.dll')],
             datas=[('inference_model', 'inference_model')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='GUI-for-paddlepaddle-OCR',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
