namespace WebAplicationAIAssistant.Models
{
    public class NeuralNetworkModel
    {
        public int Level { get; set; }
        public List<double> Weights { get; set; } = new List<double>();
        public List<TrainingExample> TrainingExamples { get; set; } = new List<TrainingExample>();
        public List<string> InputLabels { get; set; } = new List<string>();
        public string ProductName { get; set; } = "Видеоигра";
        public string ActivationFunctionName { get; set; } = "";
        public string ActivationFunctionFormula { get; set; } = "";
        public string ActivationFunctionDescription { get; set; } = "";
        public string ScientificName { get; set; } = "";
        public string Explanation { get; set; } = "";
        public bool HasBias { get; set; } = false;
        public double Bias { get; set; } = 0;
        public double Threshold { get; set; } = 0;
    }

    public class TrainingExample
    {
        public List<int> Inputs { get; set; } = new List<int>();
        public int ExpectedOutput { get; set; }
        public int ActualOutput { get; set; }
        public double Sum { get; set; }
    }

    public class LevelInfo
    {
        public int LevelNumber { get; set; }
        public string Title { get; set; } = "";
        public string Description { get; set; } = "";
        public List<string> InputLabels { get; set; } = new List<string>();
        public string ProductName { get; set; } = "";
        public int InputCount => InputLabels.Count;
        public string ActivationFunctionName { get; set; } = "";
        public string ActivationFunctionFormula { get; set; } = "";
        public string ActivationFunctionDescription { get; set; } = "";
        public string ScientificName { get; set; } = "";
        public string Explanation { get; set; } = "";
        public bool HasBias { get; set; } = false;
        public double Threshold { get; set; } = 0;
    }

    public class TrainingResult
    {
        public bool IsCorrect { get; set; }
        public List<TrainingExample> Results { get; set; } = new List<TrainingExample>();
        public string Message { get; set; } = "";
        public int CorrectCount { get; set; }
        public int TotalCount { get; set; }
    }
}
