#!/bin/bash
# Run the full experiment in manageable batches to avoid API overload
# Total: 2,304 vignettes split into batches

BATCH_SIZE=500
DELAY=1.0  # Increase if hitting rate limits
TOTAL_VIGNETTES=2304

echo "🚀 Starting Batched Experiment Run"
echo "=================================="
echo "Total vignettes: $TOTAL_VIGNETTES"
echo "Batch size: $BATCH_SIZE"
echo "Delay between queries: ${DELAY}s"
echo ""

# Calculate number of batches
NUM_BATCHES=$(( ($TOTAL_VIGNETTES + $BATCH_SIZE - 1) / $BATCH_SIZE ))

echo "Will run $NUM_BATCHES batches"
echo ""

# Run each batch
for ((i=0; i<NUM_BATCHES; i++)); do
    START=$(( i * BATCH_SIZE + 1 ))
    END=$(( (i + 1) * BATCH_SIZE ))
    
    # Don't exceed total vignettes
    if [ $END -gt $TOTAL_VIGNETTES ]; then
        END=$TOTAL_VIGNETTES
    fi
    
    BATCH_NUM=$(( i + 1 ))
    
    echo "📦 Batch $BATCH_NUM/$NUM_BATCHES: Vignettes $START-$END"
    echo "Starting at: $(date)"
    
    python llm_execution/run_experiment.py \
        --vignette-range "$START-$END" \
        --delay $DELAY \
        2>&1 | tee "logs/batch_${BATCH_NUM}_${START}-${END}.log"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Batch $BATCH_NUM failed with exit code $EXIT_CODE"
        echo "You can resume from batch $BATCH_NUM by running:"
        echo "  python llm_execution/run_experiment.py --vignette-range \"$START-$END\""
        exit $EXIT_CODE
    fi
    
    echo "✅ Batch $BATCH_NUM completed at: $(date)"
    echo ""
    
    # Optional: Add a pause between batches
    if [ $BATCH_NUM -lt $NUM_BATCHES ]; then
        echo "😴 Pausing 30 seconds between batches..."
        sleep 30
    fi
done

echo ""
echo "🎉 All batches completed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Merge results: python llm_execution/merge_results.py"
echo "2. Run analysis: python analysis/analyze_results.py analysis/results/merged_results.csv"
echo "3. Generate visualizations:"
echo "   - python visualization/visualize_results.py analysis/results/merged_results.csv"
echo "   - python visualization/visualize_by_outcome.py analysis/results/merged_results.csv"

