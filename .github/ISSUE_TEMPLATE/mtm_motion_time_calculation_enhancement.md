### Issue Title: Correct and enhance MTM motion time calculation logic

**Description:**
Update the MTM analysis notebook to fix the motion time calculation logic and improve event segmentation:

- Use a more robust motion threshold for production environments
- Implement temporal gap filling to handle intermittent detection failures
- Avoid double-counting overlapping motion events
- Provide functions/classes for improved motion analysis, including:
  - Enhanced velocity-based thresholding
  - Validation with sample data
- Add comments and usage examples in the notebook

**Acceptance criteria:**
- Demonstrate improved accuracy and reliability of motion time calculation in the notebook
- Include unit tests or validation metrics where feasible