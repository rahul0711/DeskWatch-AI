package com.example.facercognitionapp

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.facercognitionapp.databinding.ActivityLoginBinding
import com.example.facercognitionapp.model.FaceLoginRequest
import com.example.facercognitionapp.network.FaceApiClient
import com.example.facercognitionapp.ui.core.AppFooterHelper
import kotlinx.coroutines.launch

class LoginActivity : AppCompatActivity() {

    private lateinit var binding: ActivityLoginBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)
        AppFooterHelper.bind(binding.appFooter.root)
        binding.loginBtn.setOnClickListener { login() }
    }

    private fun login() {
        val username = binding.usernameInput.text.toString().trim()
        val password = binding.passwordInput.text.toString()

        if (username.isEmpty() || password.isEmpty()) {
            Toast.makeText(this, R.string.login_error_empty, Toast.LENGTH_SHORT).show()
            return
        }

        setLoading(true)
        lifecycleScope.launch {
            try {
                val response = FaceApiClient.api.login(FaceLoginRequest(username, password))
                val token = response.body()?.token
                if (!response.isSuccessful || token.isNullOrBlank()) {
                    val msg = if (response.code() == 401) {
                        getString(R.string.login_error_invalid)
                    } else {
                        getString(R.string.login_error_http, response.code())
                    }
                    Toast.makeText(this@LoginActivity, msg, Toast.LENGTH_LONG).show()
                    return@launch
                }

                FaceApiClient.saveToken(this@LoginActivity, token)
                startActivity(Intent(this@LoginActivity, MainActivity::class.java))
                finish()
            } catch (e: Exception) {
                Toast.makeText(
                    this@LoginActivity,
                    getString(R.string.error_server_unreachable),
                    Toast.LENGTH_LONG
                ).show()
            } finally {
                setLoading(false)
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        binding.loginBtn.isEnabled = !loading
        binding.loginBtn.text = getString(if (loading) R.string.login_please_wait else R.string.login_sign_in)
    }
}
