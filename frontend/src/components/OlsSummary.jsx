import React, { useEffect, useState } from 'react';
import '../styles/table.css'

const OlsSummary = (html) => {
  const [htmlSummary, setHtmlSummary] = useState('');

  if (!html) return null; 

  console.log(html)

  return (
    <div>
      <div id="ols-summary-container" dangerouslySetInnerHTML={{ __html: html.html }} />
    </div>
  );
};

export default OlsSummary;
