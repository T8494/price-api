--- a/price_backend_fastapi.py
+++ b/price_backend_fastapi.py
@@
-from typing import Optional
+from typing import Optional
+import re
@@
 def get_card_price(
@@
-    product_id = product.get("product_id")
-    product_url = f"https://www.pricecharting.com/game/{product_id}"
+    product_id = product.get("product_id")
+    # Prefer the canonical URL from the API if present (often includes set slug)
+    product_url = product.get("url") or f"https://www.pricecharting.com/game/{product_id}"
     product_name = product.get("product_name", name)
@@
-    graded_prices = product_data.get("graded_price", {})
-    price = None
-
-    for label, value in graded_prices.items():
-        if grade.upper() in label.upper():
-            try:
-                price = float(value)
-            except:
-                price = None
-            break
+    graded_prices = product_data.get("graded_price", {})
+    price = None
+
+    # --- Grade normalization helpers ---
+    def norm(s: str) -> str:
+        """Uppercase, strip spaces, dashes, and the word MINT to make matching robust."""
+        return re.sub(r"(MINT|[^A-Z0-9.])", "", s.upper())
+
+    def brand_and_number(s: str):
+        """Extract brand (PSA/BGS/CGC/etc.) and numeric part (e.g., 9, 9.5, 10)."""
+        s_up = s.upper()
+        brand_match = re.match(r"^(PSA|BGS|CGC|SGC)", s_up)
+        num_match = re.search(r"\d+(\.\d+)?", s_up)
+        return (brand_match.group(1) if brand_match else None,
+                num_match.group(0) if num_match else None)
+
+    target_norm = norm(grade)
+    target_brand, target_num = brand_and_number(grade)
+
+    for label, value in graded_prices.items():
+        label_norm = norm(label)
+        label_brand, label_num = brand_and_number(label)
+
+        # 1) Exact normalized match (handles PSA 9 / PSA-9 / PSA 9 Mint)
+        if label_norm == target_norm or label_norm.replace(".", "") == target_norm.replace(".", ""):
+            price = float(value) if value not in (None, "", "N/A") else None
+            break
+
+        # 2) Brand + number match (PSA + 9 == PSA 9 Mint)
+        if target_brand and label_brand and target_num and label_num:
+            if target_brand == label_brand and target_num == label_num:
+                price = float(value) if value not in (None, "", "N/A") else None
+                break
@@
     return {
         "name": product_name,
         "grade": grade,
         "price": price,
         "url": product_url
     }
