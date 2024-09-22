import shutil

def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, 'zip', folder_path)

# Ví dụ sử dụng
folder_path = './checkpoints/msim-mt5-luat/'
output_path = 'msim-mt5-luat'
zip_folder(folder_path, output_path)