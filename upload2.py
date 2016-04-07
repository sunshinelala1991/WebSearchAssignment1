import os
from flask import Flask, request, redirect, url_for,render_template
#from werkzeug import secure_filename
import time

UPLOAD_FOLDER = '/Users/yunwang/PycharmProjects/myFile/locationNUpload/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','mp3','wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            ###method
            ###get bird name
            ##birdname
            #print os.path.join(app.config['UPLOAD_FOLDER'], filename)

            time.sleep(20)
            return render_template('hello.html',name="Magpie")
    return render_template('hello.html')




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)