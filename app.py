from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and clean data from CSV file
data = pd.read_csv('data/laptop_pricing_dataset.csv')

# Mapping numeric categories to string values
category_mapping = {
    1: 'Gaming',
    2: 'Netbook',
    3: 'Notebook',
    4: 'Ultrabook',
    5: 'Workstation'
}

# Select relevant features and target variable
X = data[['RAM_GB', 'Storage_GB_SSD', 'Screen_Size_cm', 'CPU_frequency', 'Weight_kg']]
y = data['Category']  # Assuming 'Category' is numeric in the dataset

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

@app.route('/')
def home():
    # Convert the DataFrame to an HTML table
    data_html = data.to_html(classes='table table-striped', index=False)
    
    # Render the template and pass the table HTML
    return render_template('index.html', data_table=data_html)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form values from the request
        ram = float(request.form['ram'])
        storage = float(request.form['storage'])
        screen_size = float(request.form['screenSize'])
        cpu_frequency = float(request.form['cpuFrequency'])
        weight = float(request.form['weight'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[ram, storage, screen_size, cpu_frequency, weight]],
                                  columns=['RAM_GB', 'Storage_GB_SSD', 'Screen_Size_cm', 'CPU_frequency', 'Weight_kg'])

        # Use the model to predict the category
        predicted_category_numeric = model.predict(input_data)[0]

        # Convert the numeric prediction to the corresponding string category
        predicted_category_string = category_mapping.get(int(predicted_category_numeric), "Unknown")

        # Return the prediction as JSON
        return jsonify({'prediction': predicted_category_string})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
