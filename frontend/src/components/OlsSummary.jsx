// import { Button } from '@mui/material';
// import { useCallback, useMemo } from 'react';
// import '../styles/table.css';

// const OlsSummary = ({ html }) => {
//   const htmlSummary = useMemo(() => {
//     if (typeof html === 'string') {
//       return html;
//     }
//     if (html && typeof html.html === 'string') {
//       return html.html;
//     }
//     return '';
//   }, [html]);

//   const handleDownload = useCallback(() => {
//     if (!htmlSummary) return;

//     const blob = new Blob([htmlSummary], { type: 'text/html' });
//     const url = URL.createObjectURL(blob);

//     const link = document.createElement('a');
//     link.href = url;
//     link.download = 'regression_output_summary.html';
//     document.body.appendChild(link);
//     link.click();
//     document.body.removeChild(link);
//     URL.revokeObjectURL(url);
//   }, [htmlSummary]);

//   if (!htmlSummary) return null;

//   return (
//     <div>
//       <Button variant="outlined" size="small" onClick={handleDownload} sx={{ mb: 2 }}>
//         Download HTML
//       </Button>
//       <div id="ols-summary-container" dangerouslySetInnerHTML={{ __html: htmlSummary }} />
//     </div>
//   );
// };

// export default OlsSummary;
import { Button } from '@mui/material';
import { useMemo } from 'react';
import '../styles/table.css';

const OlsSummary = ({ html, htmlUrl }) => {
  const htmlSummary = useMemo(() => {
    if (typeof html === 'string') return html;
    if (html && typeof html.html === 'string') return html.html;
    return '';
  }, [html]);

  if (!htmlSummary) return null;

  const baseUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:5001";
  const fullDownloadUrl = htmlUrl
    ? (htmlUrl.startsWith("http") ? htmlUrl : `${baseUrl}${htmlUrl}`)
    : null;

  return (
    <div>
      {fullDownloadUrl ? (
        <Button
          variant="outlined"
          size="small"
          href={fullDownloadUrl}
          target="_blank"
          rel="noopener"
          sx={{ mb: 2 }}
        >
          Download HTML
        </Button>
      ) : null}

      {/* still display the regression summary html */}
      <div
        id="ols-summary-container"
        dangerouslySetInnerHTML={{ __html: htmlSummary }}
      />
    </div>
  );
};

export default OlsSummary;