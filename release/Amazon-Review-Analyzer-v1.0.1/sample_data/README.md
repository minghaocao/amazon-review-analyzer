# Sample Data

`demo_reviews.csv` is the synthetic, frozen Demo input distributed with Amazon Review Analyzer v1.0.0.

- It contains 15 English review rows.
- It does not contain customer personal information or live Amazon data.
- It intentionally includes data-quality edge cases such as a blank review, a blank title, duplicate Review ID values, and invalid ratings. These rows demonstrate validation, skipping, and coverage behavior.
- The frozen Demo analysis contains 14 successfully analyzed reviews and 93.3% coverage.

For a clean custom import, map at least one non-empty column to **Review Content**. Review ID, Rating, Title, Date, and Verified Purchase are optional in the interactive upload flow.
