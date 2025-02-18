import boto3
from botocore.client import Config
from botocore import UNSIGNED
import sys
import threading
import argparse

class ProgressPercentage(object):
    ''' Progress Class
    Class for calculating and displaying download progress
    '''
    def __init__(self, client, bucket, filename):
        ''' Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        '''
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        ''' Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size 
        and prints progress bar.
        '''
        # To simplify we'll assume this is hooked up to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round((float(self._seen_so_far) / float(self._size)) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length - len(str(percentage)) - 1) + str(percentage) + '%'

            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\r')
            else:
                sys.stdout.write(output + '\n')
            sys.stdout.flush()

bucket_name = 'ifn-wsl-ssl-data'

urls = {
    'train_wsl':'crops_train_seg_all_64x64_181b_augmented.hdf5',
    'test_wsl':'crops_test_seg_all_64x64_181b.hdf5',
    'train_mae':'crops_train_all_SSL.hdf5'
}

def main(datasets):

    if not datasets: #if no argument is parsed, then download all datasets
        datasets = ["train_wsl","test_wsl","train_mae"]

    
    # Create S3 client
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for k in urls.keys():
        if k in datasets:
            file_name = urls[k]

            progress = ProgressPercentage(s3, bucket_name, file_name)

            # Download the file to the current directory
            print("Downloading {}".format(file_name))
            s3.download_file(bucket_name, file_name, file_name, Callback=progress)

            print("File downloaded.")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Define which datasets to download")
    parser.add_argument(
        "datasets", 
        choices=["train_wsl","test_wsl","train_mae"], 
        nargs="+",
        help="Specify which datasets should be downloaded (train_wsl, test_wsl, train_mae)"
        )

    args = parser.parse_args()

    main(args.datasets)

