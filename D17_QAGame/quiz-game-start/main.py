
from question_model import Question
from data import question_data
from quiz_brain import QuizBrian





def AllQuestion():
    QuestionBank_list=[]
    for i in range(len(question_data)):
        QuestionBank=Question(question_data[i]["question"], question_data[i]["correct_answer"])
        QuestionBank_list.append(QuestionBank)

    return QuestionBank_list

QB=AllQuestion()
quiz=QuizBrian(QB)
quiz.next_question()


while quiz.still_has_question():
    quiz.next_question()


print('You completed the question quiz')
print(f'Your score is {quiz.score}/{quiz.question_number}')