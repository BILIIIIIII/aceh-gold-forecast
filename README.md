### Instructions

1. **Install Dependencies**

   ```
   pip install -r reqs.txt
   ```

2. **Run the Project**

   ```
   ./run.sh
   ```

3. **Access the API**

   - Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

   - Example endpoint: `http://127.0.0.1:8000/gold_prices/?start_date=2013-09-01&end_date=2020-12-31`

   - Specific date: `http://127.0.0.1:8000/gold_prices/2013-09-01`

### Notes

- Ensure that the scraper has permission to access and scrape the target website.

- Adjust the scraper if the website structure changes.

- For production, consider using a database for better performance and scalability.