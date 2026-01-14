using Microsoft.AspNetCore.Mvc;
using WebAplicationAIAssistant.Models;

namespace WebAplicationAIAssistant.Controllers
{
    public class NeuralNetworkController : Controller
    {
        public IActionResult Index()
        {
            var levels = GetLevels();
            return View(levels);
        }

        public IActionResult Train(int level)
        {
            var levelInfo = GetLevelInfo(level);
            if (levelInfo == null)
            {
                return RedirectToAction("Index");
            }

            var model = new NeuralNetworkModel
            {
                Level = level,
                InputLabels = levelInfo.InputLabels,
                ProductName = levelInfo.ProductName,
                Weights = new List<double>(new double[levelInfo.InputCount]),
                ActivationFunctionName = levelInfo.ActivationFunctionName,
                ActivationFunctionFormula = levelInfo.ActivationFunctionFormula,
                ActivationFunctionDescription = levelInfo.ActivationFunctionDescription,
                ScientificName = levelInfo.ScientificName,
                Explanation = levelInfo.Explanation,
                HasBias = levelInfo.HasBias,
                Threshold = levelInfo.Threshold
            };

            model.TrainingExamples = GenerateTrainingExamples(levelInfo);
            return View(model);
        }

        [HttpPost]
        public IActionResult Test([FromBody] TestRequest request)
        {
            var levelInfo = GetLevelInfo(request.Level);
            if (levelInfo == null)
                return BadRequest("Неверный уровень");

            var results = new List<TrainingExample>();
            var trainingExamples = GenerateTrainingExamples(levelInfo);
            int correctCount = 0;

            foreach (var example in trainingExamples)
            {
                double sum = CalculateWeightedSum(example.Inputs, request.Weights, levelInfo);
                int actualOutput = ApplyActivationFunction(sum, levelInfo);

                example.ActualOutput = actualOutput;
                example.Sum = sum;

                if (actualOutput == example.ExpectedOutput)
                    correctCount++;

                results.Add(example);
            }

            return Json(new TrainingResult
            {
                IsCorrect = correctCount == results.Count,
                Results = results,
                CorrectCount = correctCount,
                TotalCount = results.Count,
                Message = correctCount == results.Count
                    ? "Поздравляем! Вы правильно настроили нейросеть! 🎉"
                    : $"Правильных ответов: {correctCount} из {results.Count}. Попробуйте изменить веса!"
            });
        }

        // ================== УРОВНИ ==================

        private List<LevelInfo> GetLevels()
        {
            return new List<LevelInfo>
            {
                // ---------- УРОВЕНЬ 1 ----------
                new LevelInfo
                {
                    LevelNumber = 1,
                    Title = "Уровень 1: Начальный",
                    Description = "Два простых фактора",
                    ProductName = "Новая видеоигра",
                    InputLabels = new List<string>
                    {
                        "Есть деньги?",
                        "Родители разрешили?"
                    },
                    ActivationFunctionName = "Логическая разность",
                    ActivationFunctionFormula =
                        "S = X₁·W₁ − X₂·W₂\n" +
                        "y = {1, если S ≥ 0; 0, иначе}",
                    ScientificName = "Difference Function",
                    Explanation =
                        "Первый фактор положительный, второй отрицательный. " +
                        "Если денег достаточно и запрет не перевешивает — покупаем.",
                    Threshold = 0
                },

                // ---------- УРОВЕНЬ 2 ----------
                new LevelInfo
                {
                    LevelNumber = 2,
                    Title = "Уровень 2: Средний",
                    Description = "Решение по большинству факторов",
                    ProductName = "Игровая консоль",
                    InputLabels = new List<string>
                    {
                        "Есть деньги?",
                        "Разрешили родители?",
                        "Хорошие отзывы?"
                    },
                    ActivationFunctionName = "Функция большинства",
                    ActivationFunctionFormula =
                        "S = X₁·W₁ + X₂·W₂ + X₃·W₃\n" +
                        "y = {1, если S ≥ 2; 0, иначе}",
                    ScientificName = "Majority Function",
                    Explanation =
                        "Решение принимается, если минимум два из трёх факторов положительные.",
                    Threshold = 2   // ★ ИЗМЕНЕНО
                },

                // ---------- УРОВЕНЬ 3 ----------
                new LevelInfo
                {
                    LevelNumber = 3,
                    Title = "Уровень 3: Продвинутый",
                    Description = "Нелинейная логика",
                    ProductName = "Игровой ноутбук",
                    InputLabels = new List<string>
                    {
                        "Есть деньги?",
                        "Скидка есть?",
                        "Хорошие характеристики?",
                        "Есть время играть?"
                    },
                    ActivationFunctionName = "XOR + AND",
                    ActivationFunctionFormula =
                        "S = (X₁ ⊕ X₂)·W₁ + (X₃ · X₄)·W₂\n" +
                        "y = {1, если S ≥ 1; 0, иначе}",
                    ScientificName = "Non-linear Logical Function",
                    Explanation =
                        "XOR показывает, что один фактор должен быть истинным, но не оба. " +
                        "AND усиливает влияние двух одновременно истинных факторов.",
                    Threshold = 1   // ★ ИЗМЕНЕНО
                }
            };
        }

        private LevelInfo? GetLevelInfo(int level)
        {
            return GetLevels().FirstOrDefault(l => l.LevelNumber == level);
        }

        // ================== ОБУЧЕНИЕ ==================

        private List<TrainingExample> GenerateTrainingExamples(LevelInfo levelInfo)
        {
            var examples = new List<TrainingExample>();
            int combinations = (int)Math.Pow(2, levelInfo.InputLabels.Count);
            var correctWeights = GetCorrectWeights(levelInfo);

            for (int i = 0; i < combinations; i++)
            {
                var inputs = new List<int>();
                int temp = i;

                for (int j = 0; j < levelInfo.InputLabels.Count; j++)
                {
                    inputs.Add(temp % 2);
                    temp /= 2;
                }

                double sum = CalculateWeightedSumForLevel(inputs, correctWeights, levelInfo);
                int expected = ApplyActivationFunction(sum, levelInfo);

                examples.Add(new TrainingExample
                {
                    Inputs = inputs,
                    ExpectedOutput = expected
                });
            }

            return examples;
        }

        private List<double> GetCorrectWeights(LevelInfo levelInfo)
        {
            switch (levelInfo.LevelNumber)
            {
                case 1:
                    return new List<double> { 2, 1 };

                case 2: // ★ ИЗМЕНЕНО
                    return new List<double> { 1, 1, 1 };

                case 3: // ★ ИЗМЕНЕНО
                    return new List<double> { 1, 1 };

                default:
                    return new List<double>();
            }
        }

        private double CalculateWeightedSum(
            List<int> inputs,
            List<double> weights,
            LevelInfo levelInfo)
        {
            return CalculateWeightedSumForLevel(inputs, weights, levelInfo);
        }

        private double CalculateWeightedSumForLevel(
            List<int> inputs,
            List<double> weights,
            LevelInfo levelInfo)
        {
            switch (levelInfo.LevelNumber)
            {
                case 1:
                    return inputs[0] * weights[0] - inputs[1] * weights[1];

                case 2: // ★ Majority
                    double sum = 0;
                    for (int i = 0; i < inputs.Count; i++)
                        sum += inputs[i] * weights[i];
                    return sum;

                case 3: // ★ XOR + AND
                    int xor = inputs[0] ^ inputs[1];
                    int and = inputs[2] & inputs[3];
                    return xor * weights[0] + and * weights[1];

                default:
                    return 0;
            }
        }

        private int ApplyActivationFunction(double sum, LevelInfo levelInfo)
        {
            return sum >= levelInfo.Threshold ? 1 : 0;
        }
    }

    public class TestRequest
    {
        public int Level { get; set; }
        public List<double> Weights { get; set; } = new();
    }

}
