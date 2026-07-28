import path from "path";
import fs from "fs/promises";
import matter from "gray-matter";
import { titleCase } from "title-case";
import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js";

const DEFAULT_ICON = "info-circle";
const lessonsPath = path.join(process.cwd(), "lessons");

const marked = new Marked(
  markedHighlight({
    baseUrl: process.env.BASE_URL ? process.env.BASE_URL + "/" : "/",
    highlight: function (code, lang) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
    langPrefix: "hljs language-",
  })
);

function getTitle(slug, override) {
  let title = override;
  if (!title) {
    title = titleCase(slug.split("-").join(" "));
  }

  return title;
}

async function getMeta(section) {
  let meta = {};
  try {
    const file = await fs.readFile(
      path.join(lessonsPath, section, "meta.json")
    );
    meta = JSON.parse(file.toString());
  } catch (e) {
    // no meta.json, nothing to do
  }

  return meta;
}

function slugify(inputPath) {
  const pathParts = inputPath.split("-");
  const pathOrder = pathParts.shift();
  const pathSlug = pathParts.join("-");
  return {
    slug: pathSlug,
    order: pathOrder,
    title: titleCase(pathParts.join(" ")),
  };
}

export async function getLessons() {
  const dir = await fs.readdir(lessonsPath);
  const sections = [];

  for (let dirFilename of dir) {
    const dirStats = await fs.lstat(path.join(lessonsPath, dirFilename));

    if (!dirStats.isDirectory()) {
      continue;
    }

    const lessonsDir = await fs.readdir(path.join(lessonsPath, dirFilename));

    let {
      title: sectionTitle,
      order: sectionOrder,
      slug: sectionSlug,
    } = slugify(dirFilename);

    let icon = DEFAULT_ICON;

    const meta = await getMeta(dirFilename);
    if (meta.title) {
      sectionTitle = meta.title;
    }
    if (meta.icon) {
      icon = meta.icon;
    }

    // Build a map of all lessons in this section
    const lessonsMap = {};
    const allLessons = [];
    
    for (let lessonFilename of lessonsDir) {
      if (lessonFilename.slice(-3) !== ".md") {
        continue;
      }

      const filePath = path.join(lessonsPath, dirFilename, lessonFilename);

      const file = await fs.readFile(filePath);
      const { data } = matter(file.toString());
      let slug = lessonFilename.replace(/\.md$/, "");

      const slugParts = slug.split("-");
      const lessonOrder = slugParts.shift();

      slug = slugParts.join("-");

      const title = getTitle(slug, data.title);

      const lesson = {
        slug,
        fullSlug: `/lessons/${sectionSlug}/${slug}`,
        title,
        order: `${sectionOrder}${lessonOrder.toUpperCase()}`,
        path: filePath,
        description: data.description ? data.description : "",
      };
      
      lessonsMap[lessonFilename] = lesson;
      lessonsMap[slug] = lesson;
      allLessons.push(lesson);
    }

    // Check if meta has nested sections
    let subsections = [];
    if (meta.sections && Array.isArray(meta.sections)) {
      // Use the new nested structure
      for (const subsection of meta.sections) {
        const subsectionLessons = [];
        if (subsection.order && Array.isArray(subsection.order)) {
          for (const lessonSlug of subsection.order) {
            // lessonSlug could be filename (A-introduction) or just slug (introduction)
            const lesson = lessonsMap[lessonSlug] || lessonsMap[lessonSlug + ".md"];
            if (lesson) {
              subsectionLessons.push(lesson);
            }
          }
        }
        subsections.push({
          title: subsection.title || subsection.name,
          lessons: subsectionLessons,
        });
      }
    } else if (meta.order && Array.isArray(meta.order)) {
      // Use the old flat order array for backward compatibility
      const orderedLessons = [];
      const orderedSlugs = new Set();
      for (const lessonSlug of meta.order) {
        const lesson = lessonsMap[lessonSlug] || lessonsMap[lessonSlug + ".md"];
        if (lesson) {
          orderedLessons.push(lesson);
          // Track both the filename format and the slug format
          orderedSlugs.add(lessonSlug);
          orderedSlugs.add(lessonSlug + ".md");
          orderedSlugs.add(lesson.slug);
        }
      }
      // If there are lessons not in the order, add them at the end
      for (const lesson of allLessons) {
        if (!orderedSlugs.has(lesson.slug)) {
          orderedLessons.push(lesson);
        }
      }
      subsections.push({
        title: sectionTitle,
        lessons: orderedLessons,
      });
    } else {
      // No order specified, use all lessons
      subsections.push({
        title: sectionTitle,
        lessons: allLessons,
      });
    }

    sections.push({
      icon,
      title: sectionTitle,
      slug: sectionSlug,
      subsections,
      lessons: allLessons, // Keep for backward compatibility
      order: sectionOrder,
    });
  }

  return sections;
}

export async function getLesson(targetDir, targetFile) {
  const sections = await getLessons();
  
  // Find the target section
  const targetSection = sections.find((section) => section.slug === targetDir);
  if (!targetSection) {
    return false;
  }

  // Get all lessons from the section (flatten subsections if they exist)
  let allLessons = [];
  let lessonToSubsectionMap = new Map();
  
  if (targetSection.subsections && targetSection.subsections.length > 0) {
    for (const subsection of targetSection.subsections) {
      for (const subLesson of subsection.lessons) {
        allLessons.push(subLesson);
        // Map lesson slug to subsection title
        lessonToSubsectionMap.set(subLesson.slug, subsection.title);
      }
    }
  } else {
    allLessons = targetSection.lessons || [];
  }

  // Find the current lesson index
  const currentIndex = allLessons.findIndex((lesson) => {
    // Compare by slug (without the leading order character)
    const lessonSlug = lesson.slug;
    const targetSlug = targetFile;
    
    // Handle both formats: "A-introduction" vs "introduction"
    const lessonParts = lessonSlug.split("-");
    const lessonOrder = lessonParts.shift();
    const lessonBaseSlug = lessonParts.join("-");
    
    return lessonSlug === targetSlug || lessonBaseSlug === targetSlug;
  });

  if (currentIndex === -1) {
    return false;
  }

  const lesson = allLessons[currentIndex];
  const filePath = lesson.path;
  const file = await fs.readFile(filePath);
  const { data, content } = matter(file.toString());
  let html = marked.parse(content);
  
  // Rewrite image src paths to include BASE_URL for static export
  if (process.env.BASE_URL) {
    html = html.replace(/(<img[^>]*src=")\/images\//g, `$1${process.env.BASE_URL}/images/`);
  }
  
  const title = getTitle(targetFile, data.title);

  const meta = await getMeta(targetDir);
  const section = getTitle(targetDir, meta.title);
  const icon = meta.icon ? meta.icon : DEFAULT_ICON;
  
  // Get subsection title if available
  let subsectionTitle = null;
  if (targetSection.subsections && targetSection.subsections.length > 0) {
    subsectionTitle = lessonToSubsectionMap.get(lesson.slug);
  }

  let nextSlug = null;
  let prevSlug = null;

  // get next
  if (currentIndex < allLessons.length - 1) {
    const nextLesson = allLessons[currentIndex + 1];
    nextSlug = `/lessons/${targetDir}/${nextLesson.slug}`;
  } else {
    // Try to find next in the next section
    const currentSectionIndex = sections.findIndex((s) => s.slug === targetDir);
    if (currentSectionIndex < sections.length - 1) {
      const nextSection = sections[currentSectionIndex + 1];
      const nextSectionLessons = nextSection.subsections && nextSection.subsections.length > 0
        ? nextSection.subsections.flatMap((ss) => ss.lessons)
        : nextSection.lessons || [];
      if (nextSectionLessons.length > 0) {
        nextSlug = `/lessons/${nextSection.slug}/${nextSectionLessons[0].slug}`;
      }
    }
  }

  // get prev
  if (currentIndex > 0) {
    const prevLesson = allLessons[currentIndex - 1];
    prevSlug = `/lessons/${targetDir}/${prevLesson.slug}`;
  } else {
    // Try to find prev in the previous section
    const currentSectionIndex = sections.findIndex((s) => s.slug === targetDir);
    if (currentSectionIndex > 0) {
      const prevSection = sections[currentSectionIndex - 1];
      const prevSectionLessons = prevSection.subsections && prevSection.subsections.length > 0
        ? prevSection.subsections.flatMap((ss) => ss.lessons)
        : prevSection.lessons || [];
      if (prevSectionLessons.length > 0) {
        prevSlug = `/lessons/${prevSection.slug}/${prevSectionLessons[prevSectionLessons.length - 1].slug}`;
      }
    }
  }

  return {
    attributes: data,
    html,
    markdown: content,
    slug: targetFile,
    title,
    section,
    subsection: subsectionTitle,
    icon,
    filePath,
    nextSlug,
    prevSlug,
  };
}
