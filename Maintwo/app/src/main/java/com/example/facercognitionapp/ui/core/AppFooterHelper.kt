package com.example.facercognitionapp.ui.core

import android.content.Intent
import android.net.Uri
import android.view.View
import com.example.facercognitionapp.R

object AppFooterHelper {

    private const val SCRIPT_INDIA_URL = "https://scriptindia.in"

    fun bind(root: View) {
        val openLink = View.OnClickListener { view ->
            view.context.startActivity(
                Intent(Intent.ACTION_VIEW, Uri.parse(SCRIPT_INDIA_URL))
            )
        }
        root.findViewById<View>(R.id.footerLogo)?.setOnClickListener(openLink)

    }
}
