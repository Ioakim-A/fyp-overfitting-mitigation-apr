import csv
import sys
import os

def process_csv(input_file, output_file):
    """
    Process a CSV file to replace the 'id' column values with the part before the first dot.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    try:
        with open(input_file, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            
            # Check if 'id' is in the headers
            if 'id' not in reader.fieldnames:
                print("Error: CSV file does not have an 'id' column")
                return False
            
            # Prepare to write output file with the same fieldnames
            fieldnames = reader.fieldnames
            
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Process each row
                for row in reader:
                    if row['id'] and '.' in row['id']:
                        row['id'] = row['id'].split('.')[0]
                    writer.writerow(row)
                
        print(f"Successfully processed CSV. Output saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_project_details.py input.csv [output.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_processed{ext}"
    
    process_csv(input_file, output_file)
