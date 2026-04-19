# CMU MLD Faculty Scraper

抓取 CMU Machine Learning Department（MLD）`core-faculty` 页面对应的教授数据，并输出为统一 JSON 格式：

```json
{
  "name": "",
  "title": "",
  "university": "",
  "department": "",
  "homepage_url": "",
  "homepage_content": "",
  "google_scholar_url": ""
}
```

## 使用方法

```bash
python3 scrape_cmu_mld_professors.py
```

运行后会在当前目录生成：

- `cmu_mld_professors.json`

## 数据来源（仅 MLD Core Faculty）

- `https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json`