import os

db_name_list = []
rename_dict = {}
dbs = {}
directory_name = '2020-02-09'

for file in os.listdir(os.path.join('data', directory_name)):
    parts = file.split('_')
    if parts[0] not in db_name_list:
        db_name_list.append(parts[0])
    old_num = parts[1].split('.')
    new_fn = f'{parts[0]}_{int(old_num[0]):03d}' + '.csv'
    rename_dict[file] = new_fn

for rename in rename_dict:
    print(rename, rename_dict[rename])
    os.rename(os.path.join('data', directory_name, rename), os.path.join('data', directory_name, rename_dict[rename]))

for file in os.listdir(os.path.join('data', directory_name)):
    parts = file.split('_')
    current_file = open(os.path.join('data', directory_name, file), 'r', encoding='utf-8').read()
    if parts[0] in dbs:
        dbs[parts[0]] += current_file
    else:
        dbs[parts[0]] = current_file

for item in dbs:
    with open(os.path.join('data', item + '.csv'), 'w', encoding='utf-8') as outfile:
        outfile.write(dbs[item])
    print('Wrote file', item + '.csv.')
