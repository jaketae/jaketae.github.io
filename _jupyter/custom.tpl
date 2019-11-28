  {% extends 'markdown.tpl' %}</p>
<p><!-- Add Div for input area -->
  {% block input %}
  <div class="input_area" markdown="1">
  {{ super() }}
  </div>
  {% endblock input %}</p>
<p><!-- Remove indentations for output text and add div classes  -->
  {% block stream %}
  {:.output_stream}</p>

<pre><code>  {{ output.text }}</code></pre>
<p>{% endblock stream %}</p>
<p>{% block data_text %}
  {:.output_data_text}</p>

<pre><code>  {{ output.data['text/plain'] }}</code></pre>
<p>{% endblock data_text %}</p>
<p>{% block traceback_line  %}
  {:.output_traceback_line}</p>

<pre><code>  {{ line | strip_ansi }}</code></pre>
<p>{% endblock traceback_line  %}</p>
<p><!-- Tell Jekyll not to render HTML output blocks as markdown -->
  {% block data_html %}
  <div markdown="0">
  {{ output.data['text/html'] }}
  </div>
  {% endblock data_html %}