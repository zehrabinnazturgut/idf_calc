# Streamlit Paket

Bu paket, uygulamayi Streamlit Community Cloud veya benzeri Python hostlara tasimak icin hazirlanmistir.

## Icerik

- `streamlit_app.py`
- `streamlit_backend/`
- `requirements.txt`
- `.python-version`

## Streamlit Community Cloud

1. Bu paket icerigini bir GitHub reposuna yukleyin.
2. Streamlit Cloud uzerinden repo baglayin.
3. Ana dosya olarak `streamlit_app.py` secin.
4. Deploy edin.

## Yerel calistirma

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Not

Open-Meteo kota limiti devam eder. Uygulama backend cache kullanir, ancak bazi cloud ortamlarda dosya tabanli cache kalici olmayabilir.
