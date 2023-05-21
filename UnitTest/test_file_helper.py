from unittest import TestCase
import Helpers.file_helper as fh


class Test(TestCase):

    def test_library_random_variables(self):
        print("hello")
        obj = fh.FileHelper()
        data = "testing123"
        file_name1 = "test.json"

        obj.write_json_to_file(data, file_name1)

    def test_file_write(self):
        obj = fh.FileHelper()
        data = "testing123"
        file_name1 = "test.json"

        obj.write_json_to_file(data, file_name1)

    def test_create_directory(self):
        dir_path = "../DataFiles/Test"

        obj = fh.FileHelper()
        obj.create_directory(dir_path)

