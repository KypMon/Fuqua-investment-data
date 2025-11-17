// import '@fsb/fw-auth/dist/fw-auth.js';
// import { useEffect, useRef } from "react";

// export default function FwAuthWrapper({ validateUrl, url, onAuth }) {
//     console.log("FwAuthWrapper validateUrl", validateUrl, "url", "url");
//     const authRef = useRef(null);

//     // Pass validateUrl as a property to the web component
//     useEffect(() => {
//         if (authRef.current && validateUrl) {
//             authRef.current.validateUrl = validateUrl;
//         }
//     }, [validateUrl]);

//     useEffect(() => {
//         async function handleGlobalAuth() {
//             try {
//                 const response = await fetch(validateUrl, {
//                     method: "GET",
//                     mode: "cors",
//                     cache: "no-cache",
//                     credentials: "include"
//                 });
//                 if (response.ok) {
//                     const json = await response.json();
//                     const jwt = json.jwt;
//                     if (jwt) {
//                         onAuth?.(jwt);
//                     }
//                 }
//             } catch (err) {
//                 console.error("Error fetching JWT:", err);
//             }
//         }

//         document.addEventListener("_FW_AUTHENTICATED", handleGlobalAuth);
//         return () => {
//             document.removeEventListener("_FW_AUTHENTICATED", handleGlobalAuth);
//         };
//     }, [validateUrl, onAuth]);

//     return <fw-auth ref={authRef}></fw-auth>;
// }

// // import '@fsb/fw-auth/dist/fw-auth.js';
// // import { useEffect, useRef } from "react";

// // export default function FwAuthWrapper({ validateUrl, onAuth }) {
// //     console.log("FwAuthWrapper, validateUrl", validateUrl);
// //     const authRef = useRef(null);

// //     useEffect(() => {
// //         async function handleGlobalAuth() {
// //             //console.log("received _FW_AUTHENTICATED event");

// //             // Now fetch the JWT the same way the component does:
// //             try {
// //                 //console.log("validateUrl", validateUrl);
// //                 const response = await fetch(validateUrl, {
// //                     method: "GET",
// //                     mode: "cors",
// //                     cache: "no-cache",
// //                     credentials: "include"
// //                 });
// //                 if (response.ok) {
// //                     const json = await response.json();
// //                     const jwt = json.jwt;
// //                     if (jwt) {
// //                         console.log("JWT retrieved:", jwt);
// //                         onAuth?.(jwt);
// //                     } else {
// //                         console.warn("No jwt field in validate response:", json);
// //                     }
// //                 } else {
// //                     console.warn("JWT fetch failed", response.status);
// //                 }
// //             } catch (err) {
// //                 console.error("Error fetching JWT:", err);
// //             }
// //         }

// //         document.addEventListener("_FW_AUTHENTICATED", handleGlobalAuth);
// //         return () => {
// //             document.removeEventListener("_FW_AUTHENTICATED", handleGlobalAuth);
// //         };
// //     }, [validateUrl, onAuth]);

// //     return <fw-auth ref={authRef}></fw-auth>;
// // }

