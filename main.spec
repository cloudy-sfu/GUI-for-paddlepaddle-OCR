# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.pyw'],
             pathex=[],
             binaries=[],
             datas=[('paddleocr', 'paddleocr'),
                    ('mklml', 'mklml'),
                    # path/to/file -> destination_directory
                    ('languages.csv', '.'),],
             hiddenimports=['pyclipper', 'numpy.f2py', 'scipy._lib.array_api_compat.numpy.fft', 
                            'scipy.special._special_ufuncs'],
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
          name='GUI for paddlepaddle OCR',
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
               name='GUI for paddlepaddle OCR')
