package com.smartfarming.ai

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.webkit.*
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private val FILE_CHOOSER_RESULT_CODE = 1001
    private val PERMISSION_REQUEST_CODE = 2002

    // Default IP set to Wi-Fi IP 172.18.3.109:8000
    private val DEFAULT_SERVER_URL = "http://172.18.3.109:8000"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        checkPermissions()
        setupWebView()
    }

    private fun getServerUrl(): String {
        val prefs = getSharedPreferences("SmartFarmingPrefs", Context.MODE_PRIVATE)
        return prefs.getString("server_url", DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL
    }

    private fun saveServerUrl(url: String) {
        val prefs = getSharedPreferences("SmartFarmingPrefs", Context.MODE_PRIVATE)
        prefs.edit().putString("server_url", url).apply()
    }

    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )
        val missingPermissions = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missingPermissions.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
    }

    private fun setupWebView() {
        val webSettings = webView.settings
        webSettings.javaScriptEnabled = true
        webSettings.domStorageEnabled = true
        webSettings.allowFileAccess = true
        webSettings.allowContentAccess = true
        webSettings.mediaPlaybackRequiresUserGesture = false
        webSettings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW

        // Interface for JavaScript inside offline fallback page to change server IP
        webView.addJavascriptInterface(object {
            @JavascriptInterface
            fun setServerIp(newIp: String) {
                var formattedUrl = newIp.trim()
                if (!formattedUrl.startsWith("http://") && !formattedUrl.startsWith("https://")) {
                    formattedUrl = "http://$formattedUrl"
                }
                if (!formattedUrl.contains(":8000") && !formattedUrl.substringAfter("://").contains(":")) {
                    formattedUrl = "$formattedUrl:8000"
                }
                saveServerUrl(formattedUrl)
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Connecting to: $formattedUrl", Toast.LENGTH_SHORT).show()
                    webView.loadUrl(formattedUrl)
                }
            }
        }, "AndroidApp")

        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
                if (request?.isForMainFrame == true) {
                    showErrorPage(getServerUrl())
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                runOnUiThread {
                    request?.grant(request.resources)
                }
            }

            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                this@MainActivity.filePathCallback?.onReceiveValue(null)
                this@MainActivity.filePathCallback = filePathCallback

                val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                val contentSelectionIntent = Intent(Intent.ACTION_GET_CONTENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "image/*"
                }

                val chooserIntent = Intent(Intent.ACTION_CHOOSER).apply {
                    putExtra(Intent.EXTRA_INTENT, contentSelectionIntent)
                    putExtra(Intent.EXTRA_TITLE, "Select Crop Leaf Image")
                    putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(takePictureIntent))
                }

                startActivityForResult(chooserIntent, FILE_CHOOSER_RESULT_CODE)
                return true
            }
        }

        webView.loadUrl(getServerUrl())
    }

    private fun showErrorPage(failedUrl: String) {
        val errorHtml = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; text-align: center; }
                    .card { background: #1e293b; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
                    h2 { color: #ef4444; margin-top: 0; }
                    p { color: #94a3b8; font-size: 14px; line-height: 1.6; }
                    input { width: 90%; padding: 12px; margin: 12px 0; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: white; text-align: center; font-size: 16px; }
                    button { background: #22c55e; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 8px; font-weight: bold; width: 95%; margin-top: 8px; cursor: pointer; }
                    .tip { background: #334155; padding: 10px; border-radius: 8px; margin-top: 16px; font-size: 12px; color: #cbd5e1; }
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>⚠️ Connection Error</h2>
                    <p>Could not connect to SmartFarming AI server at:<br><code>$failedUrl</code></p>
                    <p>Make sure your laptop and phone are connected to the same Wi-Fi and the python server is running.</p>
                    
                    <input type="text" id="ipInput" value="$failedUrl" placeholder="http://172.18.3.109:8000">
                    <button onclick="connect()">Connect / Retry</button>
                    
                    <div class="tip">
                        💡 PC Command: <code>python -m uvicorn src.app:app --host 0.0.0.0 --port 8000</code>
                    </div>
                </div>
                <script>
                    function connect() {
                        var ip = document.getElementById('ipInput').value;
                        if(window.AndroidApp) {
                            window.AndroidApp.setServerIp(ip);
                        } else {
                            window.location.href = ip;
                        }
                    }
                </script>
            </body>
            </html>
        """.trimIndent()
        webView.loadDataWithBaseURL(null, errorHtml, "text/html", "UTF-8", null)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == FILE_CHOOSER_RESULT_CODE) {
            if (filePathCallback == null) return
            var results: Array<Uri>? = null
            if (resultCode == RESULT_OK) {
                if (data != null && data.data != null) {
                    results = arrayOf(data.data!!)
                }
            }
            filePathCallback?.onReceiveValue(results)
            filePathCallback = null
        }
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
