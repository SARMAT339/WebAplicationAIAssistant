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

            // Генерируем все возможные комбинации входных данных
            model.TrainingExamples = GenerateTrainingExamples(levelInfo);

            return View(model);
        }

        [HttpPost]
        public IActionResult Test([FromBody] TestRequest request)
        {
            var levelInfo = GetLevelInfo(request.Level);
            if (levelInfo == null)
            {
                return BadRequest("Неверный уровень");
            }

            if (request.Weights.Count != levelInfo.InputCount)
            {
                return BadRequest("Неверное количество весов");
            }

            var results = new List<TrainingExample>();
            var trainingExamples = GenerateTrainingExamples(levelInfo);
            int correctCount = 0;

            foreach (var example in trainingExamples)
            {
                // Вычисляем взвешенную сумму в зависимости от уровня
                double sum = CalculateWeightedSum(example.Inputs, request.Weights, levelInfo);
                
                // Применяем функцию активации в зависимости от уровня
                int actualOutput = ApplyActivationFunction(sum, levelInfo);

                example.ActualOutput = actualOutput;
                example.Sum = sum;

                if (example.ActualOutput == example.ExpectedOutput)
                {
                    correctCount++;
                }

                results.Add(example);
            }

            var trainingResult = new TrainingResult
            {
                IsCorrect = correctCount == results.Count,
                Results = results,
                CorrectCount = correctCount,
                TotalCount = results.Count,
                Message = correctCount == results.Count
                    ? "Поздравляем! Вы правильно настроили нейросеть! 🎉"
                    : $"Правильных ответов: {correctCount} из {results.Count}. Попробуйте изменить веса!"
            };

            return Json(trainingResult);
        }

        private List<LevelInfo> GetLevels()
        {
            return new List<LevelInfo>
            {
                new LevelInfo
                {
                    LevelNumber = 1,
                    Title = "Уровень 1: Начальный",
                    Description = "Два простых фактора",
                    ProductName = "Новая видеоигра",
                    InputLabels = new List<string>
                    {
                        "Достаточно денег?",
                        "Родители разрешили?"
                    },
                    ActivationFunctionName = "Функция баланса факторов",
                    ActivationFunctionFormula = "S = X₁·W₁ - X₂·W₂\ny = {1, если S ≥ 0; 0, если S < 0}",
                    ActivationFunctionDescription = "Функция вычисляет разность между двумя взвешенными факторами. Если баланс между положительными и отрицательными факторами склоняется в сторону покупки (S ≥ 0), мы покупаем товар.",
                    ScientificName = "Difference Function (Функция разности)",
                    Explanation = "Эта функция моделирует процесс принятия решения через сравнение двух факторов. Первый фактор (X₁) имеет положительный вес (W₁), а второй (X₂) - отрицательный вес (W₂). Когда мы вычитаем взвешенный второй фактор из первого, мы получаем баланс. Если баланс положительный или равен нулю, решение - покупать. Это учит понимать, что разные факторы могут иметь противоположное влияние на решение.",
                    HasBias = false,
                    Threshold = 0
                },
                new LevelInfo
                {
                    LevelNumber = 2,
                    Title = "Уровень 2: Средний",
                    Description = "Три фактора для решения",
                    ProductName = "Игровая консоль",
                    InputLabels = new List<string>
                    {
                        "Достаточно денег?",
                        "Родители разрешили?",
                        "Хорошая оценка у игры?"
                    },
                    ActivationFunctionName = "Функция среднего взвешенного",
                    ActivationFunctionFormula = "S = (X₁·W₁ + X₂·W₂ + X₃·W₃) / 3\ny = {1, если S ≥ 0.5; 0, если S < 0.5}",
                    ActivationFunctionDescription = "Эта функция вычисляет среднее значение трех взвешенных факторов. Решение принимается, если среднее значение факторов превышает половину (0.5). Это учит понимать, что важно учитывать все факторы в равной степени.",
                    ScientificName = "Weighted Average Function (Функция среднего взвешенного)",
                    Explanation = "В этом уровне используется функция среднего взвешенного. Она суммирует все взвешенные факторы и делит результат на количество факторов (3). Это создает нормализованное значение от 0 до 1. Если среднее значение больше или равно 0.5, это означает, что большинство факторов склоняются к покупке. Такая функция полезна, когда все факторы одинаково важны для принятия решения.",
                    HasBias = false,
                    Threshold = 0.5
                },
                new LevelInfo
                {
                    LevelNumber = 3,
                    Title = "Уровень 3: Продвинутый",
                    Description = "Четыре фактора - настоящая задача!",
                    ProductName = "Игровой ноутбук",
                    InputLabels = new List<string>
                    {
                        "Достаточно денег?",
                        "Родители разрешили?",
                        "Хорошая оценка?",
                        "Есть время играть?"
                    },
                    ActivationFunctionName = "Функция комбинированного влияния",
                    ActivationFunctionFormula = "S = X₁·W₁ + X₂·W₂ + (X₃·W₃)·(X₄·W₄) - 0.5\ny = {1, если S ≥ 0; 0, если S < 0}",
                    ActivationFunctionDescription = "Эта функция комбинирует линейные и нелинейные зависимости. Первые два фактора суммируются линейно, а последние два перемножаются, что создает синергетический эффект - оба должны быть положительными одновременно. Вычитание 0.5 добавляет порог сдержанности.",
                    ScientificName = "Mixed Linear-Quadratic Function (Смешанная линейно-квадратичная функция)",
                    Explanation = "В этом уровне используется более сложная функция, которая сочетает линейную и нелинейную зависимости. Первые два фактора (X₁, X₂) влияют линейно - их вклад просто складывается. А факторы X₃ и X₄ перемножаются - это означает, что их влияние усиливается, когда оба положительны одновременно (синергия), и ослабевает, если хотя бы один отрицателен. Вычитание 0.5 создает базовый уровень сдержанности - даже при положительных факторах нужно достичь определенного порога. Такие функции используются в реальных нейросетях для моделирования сложных зависимостей между факторами.",
                    HasBias = false,
                    Threshold = 0
                }
            };
        }

        private LevelInfo? GetLevelInfo(int level)
        {
            return GetLevels().FirstOrDefault(l => l.LevelNumber == level);
        }

        private List<TrainingExample> GenerateTrainingExamples(LevelInfo levelInfo)
        {
            var examples = new List<TrainingExample>();
            int combinations = (int)Math.Pow(2, levelInfo.InputCount);

            // Определяем "правильные" веса для каждого уровня для генерации expectedOutput
            List<double> correctWeights = GetCorrectWeights(levelInfo);

            for (int i = 0; i < combinations; i++)
            {
                var inputs = new List<int>();
                int temp = i;

                for (int j = 0; j < levelInfo.InputCount; j++)
                {
                    inputs.Add(temp % 2);
                    temp /= 2;
                }

                // Вычисляем expectedOutput используя правильную функцию с правильными весами
                double sum = CalculateWeightedSumForLevel(inputs, correctWeights, levelInfo);
                int expectedOutput = ApplyActivationFunction(sum, levelInfo);

                examples.Add(new TrainingExample
                {
                    Inputs = inputs,
                    ExpectedOutput = expectedOutput
                });
            }

            return examples;
        }

        private List<double> GetCorrectWeights(LevelInfo levelInfo)
        {
            // Правильные веса для каждого уровня, которые создают желаемое поведение
            switch (levelInfo.LevelNumber)
            {
                case 1:
                    // Уровень 1: S = X₁·W₁ - X₂·W₂
                    // Правильные веса: W₁ = 2, W₂ = 1 (деньги важнее разрешения)
                    return new List<double> { 2.0, 1.0 };
                case 2:
                    // Уровень 2: S = (X₁·W₁ + X₂·W₂ + X₃·W₃) / 3
                    // Правильные веса: все равны 1.5
                    return new List<double> { 1.5, 1.5, 1.5 };
                case 3:
                    // Уровень 3: S = X₁·W₁ + X₂·W₂ + (X₃·W₃)·(X₄·W₄) - 0.5
                    // Правильные веса: W₁ = 1, W₂ = 1, W₃ = 1, W₄ = 1
                    return new List<double> { 1.0, 1.0, 1.0, 1.0 };
                default:
                    return new List<double>(new double[levelInfo.InputCount]);
            }
        }

        private double CalculateWeightedSum(List<int> inputs, List<double> weights, LevelInfo levelInfo)
        {
            return CalculateWeightedSumForLevel(inputs, weights, levelInfo);
        }

        private double CalculateWeightedSumForLevel(List<int> inputs, List<double> weights, LevelInfo levelInfo)
        {
            switch (levelInfo.LevelNumber)
            {
                case 1:
                    // Уровень 1: S = X₁·W₁ - X₂·W₂ (разность)
                    return inputs[0] * weights[0] - inputs[1] * weights[1];
                
                case 2:
                    // Уровень 2: S = (X₁·W₁ + X₂·W₂ + X₃·W₃) / 3 (среднее)
                    double sum2 = 0;
                    for (int i = 0; i < inputs.Count; i++)
                    {
                        sum2 += inputs[i] * weights[i];
                    }
                    return sum2 / 3.0;
                
                case 3:
                    // Уровень 3: S = X₁·W₁ + X₂·W₂ + (X₃·W₃)·(X₄·W₄) - 0.5
                    double sum3 = inputs[0] * weights[0] + inputs[1] * weights[1];
                    sum3 += (inputs[2] * weights[2]) * (inputs[3] * weights[3]);
                    return sum3 - 0.5;
                
                default:
                    return 0;
            }
        }

        private int ApplyActivationFunction(double sum, LevelInfo levelInfo)
        {
            // Применяем пороговую функцию активации
            return sum >= levelInfo.Threshold ? 1 : 0;
        }
    }

    public class TestRequest
    {
        public int Level { get; set; }
        public List<double> Weights { get; set; } = new List<double>();
    }
}
