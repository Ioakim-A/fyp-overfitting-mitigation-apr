import csv
import sys

def combine_csv_columns(file1, file2, output_file):
    with open(file1, newline='') as f1, open(file2, newline='') as f2, open(output_file, 'w', newline='') as fout:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(fout)

        # Read headers
        header1 = next(reader1)
        header2 = next(reader2)
        # Combine headers and rename columns 7 and 8
        combined_header = header1[:6] + ["RS-85", "RS-95"]
        writer.writerow(combined_header)

        for row1, row2 in zip(reader1, reader2):
            # Combine first 6 columns (assumed identical), then 7th from each file
            combined_row = row1[:6] + [row1[6], row2[6]]
            writer.writerow(combined_row)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_rs_columns.py <input1.csv> <input2.csv> <output.csv>")
        sys.exit(1)
    combine_csv_columns(sys.argv[1], sys.argv[2], sys.argv[3])