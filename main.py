import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QHBoxLayout, QLabel, QSlider, QMessageBox
)
from PyQt6.QtCore import Qt, QDateTime, QPointF
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QDateTimeAxis, QValueAxis, QScatterSeries
from PyQt6.QtGui import QPainter, QColor
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # For auto ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime  # Ensuring datetime is imported
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class ForecastingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Income Forecasting Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data attributes
        self.df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.grouped_df = pd.DataFrame()
        self.forecast_df = pd.DataFrame()
        self.current_model = None
        
        # Setup UI
        self.setup_ui()
        # Apply dark theme
        self.apply_dark_theme()
    
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout()
        
        # Load CSV button
        self.load_button = QPushButton("Load Income Data")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)
        
        # Chart placeholder
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Forecast controls
        forecast_layout = QHBoxLayout()
        
        # Slider for forecasting period
        self.slider_label = QLabel("Forecast Years: 20")
        self.forecast_slider = QSlider(Qt.Orientation.Horizontal)
        self.forecast_slider.setMinimum(1)
        self.forecast_slider.setMaximum(100)
        self.forecast_slider.setValue(20)
        self.forecast_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.forecast_slider.setTickInterval(1)
        self.forecast_slider.valueChanged.connect(self.slider_changed)
        forecast_layout.addWidget(self.slider_label)
        forecast_layout.addWidget(self.forecast_slider)
        
        layout.addLayout(forecast_layout)
        
        # Forecast buttons
        self.arima_button = QPushButton("Forecast with ARIMA")
        self.arima_button.clicked.connect(self.forecast_arima)
        
        self.prophet_button = QPushButton("Forecast with Prophet")
        self.prophet_button.clicked.connect(self.forecast_prophet)
        
        self.linear_button = QPushButton("Forecast with Linear Regression")
        self.linear_button.clicked.connect(self.forecast_linear)
        
        forecast_buttons_layout = QHBoxLayout()
        forecast_buttons_layout.addWidget(self.arima_button)
        forecast_buttons_layout.addWidget(self.prophet_button)
        forecast_buttons_layout.addWidget(self.linear_button)
        
        layout.addLayout(forecast_buttons_layout)
        
        # Export to Excel button
        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_to_excel)
        layout.addWidget(self.export_button)
        
        # Set main widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def apply_dark_theme(self):
        # Define dark theme styles using QSS
        dark_stylesheet = """
        /* Main Window */
        QMainWindow {
            background-color: #2b2b2b;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #3c3f41;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        
        QPushButton:hover {
            background-color: #4b4b4b;
        }
        
        QPushButton:pressed {
            background-color: #5c5c5c;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #555555;
            margin: 2px 0;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background: #ffffff;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 3px;
        }

        QSlider::handle:horizontal:hover {
            background: #dddddd;
        }

        QSlider::handle:horizontal:pressed {
            background: #cccccc;
        }
        
        /* Labels */
        QLabel {
            color: #ffffff;
            font-size: 14px;
        }
        
        /* Chart View */
        QChartView {
            background-color: #2b2b2b;
        }
        
        /* QMessageBox */
        QMessageBox {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        /* Gradient Background for the Main Window */
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e1e1e, stop:1 #121212);
        }
        """
        self.setStyleSheet(dark_stylesheet)
    
    def load_csv(self):
        # Define the default directory
        default_directory = "D:/Libraries/Documents/Echo Sphere One/Main/gui/data_processing/exported_data/"
        
        # Open file dialog with the default directory
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open CSV", 
            default_directory,  # Set default directory here
            "CSV Files (*.csv)"
        )
        if file_path:
            try:
                # Read CSV with specified date format
                self.df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
                
                # Filter Description field for specific companies
                companies = ["gl Group LTD","JP102596A DWP JSA","Maddison Group","Megpay Limited","Staffline Recruitm","XACT D S LTD","Gl Group LTD","Am 2 Pm Rcrtmnt Sl","JP102596A DWP UC","Pertemps Limited","Cadent Gas Limited","NUTMEG.COM"]  # Add your company names here
                self.filtered_df = self.df[self.df['Description'].isin(companies)]
                
                if self.filtered_df.empty:
                    QMessageBox.warning(self, "No Data", "No transactions found for the specified companies.")
                    return
                
                # Group by month and year, summing 'Paid in'
                self.filtered_df['YearMonth'] = self.filtered_df['Date'].dt.to_period('M')
                self.grouped_df = self.filtered_df.groupby('YearMonth')['Paid in'].sum().reset_index()
                self.grouped_df['YearMonth'] = self.grouped_df['YearMonth'].dt.to_timestamp()
                self.grouped_df = self.grouped_df.sort_values('YearMonth')
                
                # Plot the data
                self.plot_data()
                
                # Clear any existing forecast
                self.forecast_df = pd.DataFrame()
                self.current_model = None
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading CSV: {e}")
    
    def plot_data(self):
        if self.grouped_df.empty:
            return
        
        # Create a line series for actual data
        self.series_actual = QLineSeries()
        self.series_actual.setName("Actual Paid In")
        for index, row in self.grouped_df.iterrows():
            timestamp = row['YearMonth']
            value = row['Paid in']
            self.series_actual.append(timestamp.timestamp() * 1000, value)  # QDateTime uses milliseconds
        
        # Create a scatter series for tooltips
        self.scatter_series = QScatterSeries()
        self.scatter_series.setName("Actual Data Points")
        self.scatter_series.setMarkerSize(5.0)
        self.scatter_series.setColor(QColor("#00ff99"))  # Teal color for actual data
        for index, row in self.grouped_df.iterrows():
            timestamp = row['YearMonth']
            value = row['Paid in']
            point = QPointF(timestamp.timestamp() * 1000, value)
            self.scatter_series.append(point)
        
        # Create chart
        self.chart = QChart()
        self.chart.addSeries(self.series_actual)
        self.chart.addSeries(self.scatter_series)
        self.chart.setTitle("Paid In Over Time")
        self.chart.legend().setVisible(False)
        
        # Create axes
        self.axis_x = QDateTimeAxis()
        self.axis_x.setFormat("MMM yyyy")
        self.axis_x.setTitleText("Date")
        self.axis_x.setTickCount(12)
        self.axis_x.setLabelsAngle(-90)  # Rotate labels vertically
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series_actual.attachAxis(self.axis_x)
        self.scatter_series.attachAxis(self.axis_x)
        
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Paid In Sum")
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series_actual.attachAxis(self.axis_y)
        self.scatter_series.attachAxis(self.axis_y)
        
        # Connect hovered signal for tooltips
        self.scatter_series.hovered.connect(self.show_tooltip)
        
        self.chart_view.setChart(self.chart)
    
    def show_tooltip(self, point, state):
        if state:
            # Find the corresponding data point
            timestamp_ms = point.x()
            value = point.y()
            timestamp = QDateTime.fromMSecsSinceEpoch(int(timestamp_ms)).toPyDateTime()
            tooltip_text = f"{timestamp.strftime('%B %Y')}: {value:.2f}"
            self.chart_view.setToolTip(tooltip_text)
        else:
            self.chart_view.setToolTip("")
    
    def slider_changed(self, value):
        self.slider_label.setText(f"Forecast Years: {value}")
    
    def forecast_arima(self):
        if self.grouped_df.empty:
            QMessageBox.warning(self, "No Data", "Please load and filter the CSV data first.")
            return
        
        try:
            # Prepare the data
            ts = self.grouped_df.set_index('YearMonth')['Paid in']
            
            # Fit auto ARIMA model to determine best parameters with seasonality
            model = auto_arima(ts, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
            model_fit = model.fit(ts)
            
            # Forecast
            forecast_years = self.forecast_slider.value()
            steps = forecast_years * 12  # Monthly steps
            forecast = model_fit.predict(n_periods=steps)
            
            # Check if forecast is empty
            if len(forecast) == 0:
                QMessageBox.warning(self, "Forecast Error", "ARIMA model failed to generate forecast.")
                return
            
            # Create forecast dataframe
            last_date = self.grouped_df['YearMonth'].iloc[-1]
            forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=steps, freq='M')
            self.forecast_df = pd.DataFrame({
                'YearMonth': forecast_dates,
                'Paid in': forecast,
                'Model': 'ARIMA'
            })
            self.current_model = 'ARIMA'
            
            # Plot forecast
            self.plot_forecast()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ARIMA Forecasting failed: {e}")
    
    def forecast_prophet(self):
        if self.grouped_df.empty:
            QMessageBox.warning(self, "No Data", "Please load and filter the CSV data first.")
            return
        
        try:
            # Prepare data for Prophet
            df_prophet = self.grouped_df.rename(columns={'YearMonth': 'ds', 'Paid in': 'y'})
            
            # Fit Prophet model
            model = Prophet()
            model.fit(df_prophet)
            
            # Forecast
            forecast_years = self.forecast_slider.value()
            future = model.make_future_dataframe(periods=forecast_years*12, freq='M')
            forecast = model.predict(future)
            
            # Extract forecasted values
            forecast_values = forecast[['ds', 'yhat']].tail(forecast_years*12)
            self.forecast_df = forecast_values.rename(columns={'ds': 'YearMonth', 'yhat': 'Paid in'})
            self.forecast_df['Model'] = 'Prophet'
            self.current_model = 'Prophet'
            
            # Plot forecast
            self.plot_forecast()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prophet Forecasting failed: {e}")
    
    def forecast_linear(self):
        if self.grouped_df.empty:
            QMessageBox.warning(self, "No Data", "Please load and filter the CSV data first.")
            return
        
        try:
            # Prepare data
            df_linear = self.grouped_df.copy()
            df_linear['Timestamp'] = df_linear['YearMonth'].map(datetime.toordinal)
            X = df_linear['Timestamp'].values.reshape(-1, 1)
            y = df_linear['Paid in'].values
            
            # Fit Linear Regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast
            forecast_years = self.forecast_slider.value()
            steps = forecast_years * 12
            last_timestamp = df_linear['Timestamp'].iloc[-1]
            future_timestamps = np.array([last_timestamp + i for i in range(1, steps + 1)]).reshape(-1, 1)
            forecast_values = model.predict(future_timestamps)
            
            # Create forecast dataframe
            last_date = self.grouped_df['YearMonth'].iloc[-1]
            forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=steps, freq='M')
            self.forecast_df = pd.DataFrame({
                'YearMonth': forecast_dates,
                'Paid in': forecast_values,
                'Model': 'Linear Regression'
            })
            self.current_model = 'Linear Regression'
            
            # Plot forecast
            self.plot_forecast()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Linear Regression Forecasting failed: {e}")
    
    def plot_forecast(self):
        if self.forecast_df.empty:
            return
        
        # Remove existing forecast series if any
        if hasattr(self, 'series_forecast'):
            self.chart.removeSeries(self.series_forecast)
        if hasattr(self, 'scatter_forecast'):
            self.chart.removeSeries(self.scatter_forecast)
        
        # Create a line series for forecast data
        self.series_forecast = QLineSeries()
        self.series_forecast.setName(f"Forecasted Paid In ({self.current_model})")
        self.series_forecast.setColor(QColor("#ff5555"))  # Red color for forecasted data
        for index, row in self.forecast_df.iterrows():
            timestamp = row['YearMonth']
            value = row['Paid in']
            self.series_forecast.append(timestamp.timestamp() * 1000, value)
        
        # Create a scatter series for forecast tooltips
        self.scatter_forecast = QScatterSeries()
        self.scatter_forecast.setName("Forecasted Data Points")
        self.scatter_forecast.setMarkerSize(5.0)
        self.scatter_forecast.setColor(QColor("#ff5555"))  # Red color for forecasted data
        for index, row in self.forecast_df.iterrows():
            timestamp = row['YearMonth']
            value = row['Paid in']
            point = QPointF(timestamp.timestamp() * 1000, value)
            self.scatter_forecast.append(point)
        
        # Add forecast series to chart
        self.chart.addSeries(self.series_forecast)
        self.chart.addSeries(self.scatter_forecast)
        self.series_forecast.attachAxis(self.axis_x)
        self.series_forecast.attachAxis(self.axis_y)
        self.scatter_forecast.attachAxis(self.axis_x)
        self.scatter_forecast.attachAxis(self.axis_y)
        
        # Connect hovered signal for forecast tooltips
        self.scatter_forecast.hovered.connect(self.show_forecast_tooltip)
        
        # Adjust the X-axis to include forecasted dates
        all_dates = pd.concat([self.grouped_df['YearMonth'], self.forecast_df['YearMonth']])
        self.axis_x.setRange(all_dates.min(), all_dates.max())
        
        # Optionally adjust Y-axis if forecast exceeds current range
        max_paid_in = max(self.grouped_df['Paid in'].max(), self.forecast_df['Paid in'].max())
        self.axis_y.setMax(max_paid_in * 1.1)  # Add 10% padding
        
        # Update the chart title
        self.chart.setTitle(f"Paid In Over Time with {self.current_model} Forecast")
        
        # Refresh the chart view
        self.chart_view.setChart(self.chart)
    
    def show_forecast_tooltip(self, point, state):
        if state:
            # Find the corresponding data point
            timestamp_ms = point.x()
            value = point.y()
            timestamp = QDateTime.fromMSecsSinceEpoch(int(timestamp_ms)).toPyDateTime()
            tooltip_text = f"{timestamp.strftime('%B %Y')}: {value:.2f}"
            self.chart_view.setToolTip(tooltip_text)
        else:
            self.chart_view.setToolTip("")
    
    def export_to_excel(self):
        if self.grouped_df.empty:
            QMessageBox.warning(self, "No Data", "No historical data to export.")
            return
        
        if self.forecast_df.empty:
            QMessageBox.warning(self, "No Forecast", "No forecast data to export.")
            return
        
        try:
            # Open file dialog to choose save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Excel File", 
                "", 
                "Excel Files (*.xlsx)"
            )
            if file_path:
                # Ensure the file has .xlsx extension
                if not file_path.lower().endswith('.xlsx'):
                    file_path += '.xlsx'
                
                # Create a Pandas Excel writer using openpyxl
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Write historical data
                    self.grouped_df.to_excel(writer, sheet_name='Historical Data', index=False)
                    
                    # Write forecasted data
                    self.forecast_df.to_excel(writer, sheet_name='Forecasted Data', index=False)
                
                QMessageBox.information(self, "Success", f"Data successfully exported to {file_path}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data to Excel: {e}")


def main():
    app = QApplication(sys.argv)
    window = ForecastingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
