import sys


def error_msg_detail(error, error_detail:sys):
    '''
    This function return the detailed error message
    '''
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = f'Error occured in python script name [{file_name}], line number [{line_no}] with the following error [{error}]'
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
       super().__init__(error) 
       self.error_message_detail=error_msg_detail(error=error, error_detail=error_detail)

    def __str__(self):
        return self.error_message_detail
    


# if __name__=='__main__':
#     try:
#         print('Exception testing started!')
#         x = int(input('Enter some age:'))
#         if x<0 or x>100:
#             raise Exception('Invalid age entered!')
#         print(f'Age entered is {x}')

#         # a = 5/0
#     except Exception as e:
#         custom_exception = CustomException(e, sys)
#         print(custom_exception)
        
        
        
