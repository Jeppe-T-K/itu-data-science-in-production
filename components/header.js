import { useContext } from "react";
import Link from "next/link";
import { Context as HeaderContext } from "../context/headerContext";
import { Context as CourseContext } from "../context/courseInfoContext";

export default function Header(props) {
  const [{ section: contextSection, subsection: contextSubsection, title: contextTitle, icon }] = useContext(HeaderContext);
  const { frontendMastersLink } = useContext(CourseContext);
  
  // Use props if available (for server-side rendering), otherwise fall back to context
  const section = props.section || contextSection;
  const subsection = props.subsection || contextSubsection;
  const title = props.title || contextTitle;
  
  // Build breadcrumb path
  const breadcrumbParts = [];
  if (section) {
    breadcrumbParts.push(section);
    if (subsection) {
      breadcrumbParts.push(subsection);
    }
    breadcrumbParts.push(title);
  }
  
  return (
    <header className="navbar">
      <h1 className="navbar-brand">
        <Link href="/">{props.courseTitle || props.title}</Link>
      </h1>
      <div className="navbar-info">
        {frontendMastersLink ? (
          <a href={frontendMastersLink} className="cta-btn">
            Watch on Frontend Masters
          </a>
        ) : null}
        {breadcrumbParts.length > 0 ? (
          <h2>
            {breadcrumbParts.map((part, index) => (
              <span key={index}>
                {part}
                {index < breadcrumbParts.length - 1 && (
                  <span className="breadcrumb-separator">
                    <i className="fas fa-chevron-right" />
                  </span>
                )}
              </span>
            ))}
          </h2>
        ) : null}
      </div>
    </header>
  );
}
