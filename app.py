from flask import Flask, jsonify, request
from flask_cors import CORS
from pdfextract import BatchExtractor
import asyncio
import os
import shutil
import uuid

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/extract', methods=['POST'])
def extract_data():
    try:
        # Assuming files are sent as multipart form-data with 'files' as the k
       
        files = request.files.getlist('files')
        
        # Save the files to a temporary folder or as needed
        temp_folder = 'temp_files'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
  
        
        file_paths = []
        for idx, file in enumerate(files):
            print(file)
            file_path = os.path.join(temp_folder, f'file_{uuid.uuid4()}.pdf')
            file.save(file_path)
            file_paths.append(file_path)
        print(os.listdir(temp_folder))

        # Initialize BatchExtractor
        extractor = BatchExtractor(temp_folder)
        
        # Extract SMILES data
        smiles_data = asyncio.run(extractor.combine())
        # Delete temporary folders 
        shutil.rmtree(temp_folder)
        shutil.rmtree('SMILES')

        return jsonify({
            'message': 'Success',
            'data': smiles_data
        }), 200

    except Exception as e:
        return jsonify({
            'message': f'Error: {str(e)}'
        }), 500





if __name__ == '__main__':
    app.run(debug=True)
