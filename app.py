import sys
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.exception import CustomException
from src.pipeline.predict_pipeline import UserInputCompileProjectData, UserInputResourcetData, UserInputCompileResourcetData, PredictOnUserInput



app = Flask(__name__)


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for predict page
@app.route('/predict', methods=['POST'])
def predict_proj_approval():

    try:
        # Capture user provided inputs for project
        project_data = {
            'teacher_prefix': request.form['teacher_prefix'],
            'school_state': request.form['school_state'],
            'project_grade_category': request.form['project_grade_category'],
            'project_subject_categories': request.form.getlist('project_subject_categories[]'),
            'project_title': request.form['project_title'],
            'project_essay_1': request.form['project_essay_1'],
            'project_essay_2': request.form['project_essay_2'],
            'project_resource_summary': request.form['project_resource_summary']
        }

        print_sep_len = 100

        # Fixing the format for the input: project_subject_categories
        print(project_data)
        print('-'*print_sep_len)
        project_data['project_subject_categories'] = (', ').join(project_data['project_subject_categories'])
        print(project_data)
        print('-'*print_sep_len)

        # Compile project dataframe
        compile_proj_data_obj = UserInputCompileProjectData(**project_data)
        df_predict = compile_proj_data_obj.prepare_df_for_predict()


        # Capture user provided inputs for project resources
        resource_descriptions = request.form.getlist('resource_description[]')
        resource_quantities = request.form.getlist('resource_quantity[]')
        resource_prices = request.form.getlist('resource_price[]')
        
        # Compile project_resource dataframe 
        compile_proj_res_obj = UserInputCompileResourcetData()
        for res_description, res_quantity, res_price in zip(resource_descriptions, resource_quantities, resource_prices):

            resource_info_obj = UserInputResourcetData(res_description=res_description, res_quantity=res_quantity, res_price=res_price)
            resource_info_dict = resource_info_obj.to_dict()

            # Add resource for the project
            compile_proj_res_obj.append_resource(resource_info_dict)

        df_predict_res = compile_proj_res_obj.prepare_df_for_predict()


        
        print(df_predict)
        print(df_predict.shape)
        print(df_predict.info())
        print('-'*print_sep_len)
        print(df_predict_res)
        print(df_predict_res.shape)
        print(df_predict_res.info())
        print('-'*print_sep_len)


        predict_obj = PredictOnUserInput()
        pred_msg, pred_class = predict_obj.predict(predict_df=df_predict, predict_df_res=df_predict_res)

        # return str(pred_output)
        # return render_template('index.html', result=pred_output)
        return render_template('index.html', message=pred_msg, message_class=pred_class)


    except Exception as e:
        custom_exception = CustomException(e, sys)
        print(custom_exception)


if __name__=="__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0") 
















