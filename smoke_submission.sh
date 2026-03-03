#!/bin/bash
# smoke_submission.sh - Prepara e testa uma smoke submission

echo "🚀 Preparando smoke submission para trilha phonetic"

# 1. Limpar submissões anteriores
echo "📁 Limpando arquivos antigos..."
rm -f submission/submission.zip submission/submission.jsonl

# 2. Verificar código
echo "🔍 Verificando main.py..."
if [ ! -f submission_src/main.py ]; then
    echo "❌ main.py não encontrado!"
    exit 1
fi

# 3. Verificar caminho de saída
if grep -q "/code_execution/predictions.jsonl" submission_src/main.py; then
    echo "❌ Caminho de saída incorreto! Deve ser /code_execution/submission/submission.jsonl"
    exit 1
fi

# 4. Testar localmente com dados de exemplo
echo "🧪 Testando localmente com dados de exemplo..."
just pack-submission
just track=phonetic run

# 5. Mostrar resultado
echo "📊 Resultado do teste local:"
cat submission/submission.jsonl
echo ""

# 6. Verificar tamanho do arquivo
if [ -f submission/submission.jsonl ]; then
    SIZE=$(wc -l < submission/submission.jsonl)
    echo "📈 Número de predições: $SIZE"
else
    echo "❌ Arquivo submission.jsonl não foi criado!"
    exit 1
fi

# 7. Verificar formato das primeiras linhas
echo "🔍 Verificando formato das primeiras predições:"
head -3 submission/submission.jsonl | python -m json.tool

# 8. Instruções finais
echo ""
echo "✅ Smoke submission preparada!"
echo ""
echo "📦 Arquivo para upload: $(pwd)/submission/submission.zip"
echo "   Tamanho: $(ls -lh submission/submission.zip | awk '{print $5}')"
echo ""
echo "🌐 Para submeter no site:"
echo "   1. Acesse https://www.drivendata.org/competitions/"
echo "   2. Vá para a aba 'Submit' da competição"
echo "   3. Faça upload do arquivo: submission/submission.zip"
echo "   4. Marque como 'smoke test' (se disponível)"
echo "   5. Clique em Submit e acompanhe o status"

