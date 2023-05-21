import json
import os
import jsonpickle

class FileHelper:

    file_base_path = "DataFiles/"
    # "../DataFiles/"

    def write_json_to_file(self, data, file_name):

        json_data = jsonpickle.encode(data)
        with open(self.file_base_path + file_name,
                  'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    def read_json_from_file(self, file_name):
        with open(self.file_base_path + file_name, ) as f:
            data = json.load(f)

        obj = jsonpickle.decode(data)
        return obj

    def create_directory(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)



