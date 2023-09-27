from shiny import App, render, ui, reactive
import numpy as np
import joblib
import pandas as pd
import optbinning
from sklearn.preprocessing import MinMaxScaler

model_path = 'cscore/modelo_scorecard(300-850).pkl'

result = ''

model = joblib.load(model_path)

app_ui = ui.page_fluid(
    ui.markdown(
        """
        # Predicción del Score Crediticio
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(

            #### DEFINICIÓN DE INPUTS PARA LAS VARIABLES

            ui.input_numeric("person_income", "Ingresos ($)", 69900),
            ui.input_select("person_home_ownership", "Estado de propiedad",
                            {0: "MORTGAGE", 1: "RENT", 2: "OWN", 3: "OTHER"}),
            ui.input_numeric("person_emp_length", "Tiempo como empleado (años)", 9.0),
            ui.input_select("loan_intent", "Objetivo del préstamo",
                            {0: "MEDICINAL", 1: "DEBTCONSOLIDATION", 2: "PERSONAL", 3: "VENTURE", 4: "HOMEIMPROVEMENT",
                             5: "EDUCATIONAL"}),
            ui.input_select("loan_grade", "Grado del préstamo", {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}),
            ui.input_numeric("loan_amnt", "Cantidad de préstamo ($)", 5050),
            ui.input_numeric("loan_int_rate", "Interés del préstamo (%)", 5.42),
            ui.input_slider("loan_percent_income", "Porcentaje de ingreso (0.07 = 7%)", value=0.07, min=0, max=1,
                            step=0.1),
            ui.input_select("cb_person_default_on_file", "¿Ha hecho default en alguno de sus préstamos?",
                            {0: "N", 1: "Y"}),

            ui.input_action_button('btn', 'Calcular')),

        ui.panel_main(
            ui.markdown(
                """
                Predicción del modelo


                """
            ),
            ui.output_text_verbatim("txt", placeholder=True),

            ui.markdown(
                """
                <div>
                <p style = 'text-align:center;'>
                    <img src="data:image/jpeg;base64,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">
                </p>



                <div>
                <p>
                    <a href="https://github.com/AmilderO/credit_score/blob/main/FDA_T1_ReporteT%C3%A9cnico_CreditRisk.pdf">Ver el informe</a>
                </p>
                </div>
                </div>
                """
            )
        )
    )
)


def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.btn)
    def _():
        testDataframe = pd.DataFrame([[input.person_income(), input.person_home_ownership(), input.person_emp_length(),
                                       input.loan_intent(), input.loan_grade(), input.loan_amnt(),
                                       input.loan_int_rate(),
                                       input.loan_percent_income(), input.cb_person_default_on_file()]],
                                     columns=["person_income", "person_home_ownership", "person_emp_length",
                                              "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
                                              "loan_percent_income", "cb_person_default_on_file"])

        result = model.score(testDataframe)
        print("resultado es:", result)

        @output
        @render.text
        def txt():
            return f"su score crediticio es: {result}"


app = App(app_ui, server)
