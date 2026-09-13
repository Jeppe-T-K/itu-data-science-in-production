import fs from "fs/promises";
import path from "path";
import { getLessons } from "../data/lesson.js";
import puppeteer from "puppeteer";
import * as cheerio from "cheerio";
import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";
import hljs from "highlight.js";

// Create marked instance for PDF generation
const marked = new Marked(
  markedHighlight({
    highlight: function (code, lang) {
      const language = hljs.getLanguage(lang) ? lang : "plaintext";
      return hljs.highlight(code, { language }).value;
    },
    langPrefix: "hljs language-",
  })
);

// Map of lesson directory slugs to image directory names
const LESSON_IMAGE_MAP = {
  "data-and-model-versioning": "data-and-model-versioning",
  "containers-and-docker": "containers-and-docker",
  "mlops-projects-overview": "mlops-projects-overview",
  "model-experimentation": "model-experimentation",
};

/**
 * Expands all <details> sections in HTML to make their content visible
 * Handles nested details sections and preserves styling from summary elements
 */
function expandDetails(html) {
  const $ = cheerio.load(html, { xmlMode: false });

  // Process details from innermost to outermost
  function processDetailsRecursive($element) {
    const $detailsElements = $element.find("details");
    
    if ($detailsElements.length === 0) {
      return;
    }
    
    // First, process nested details in each details element
    $detailsElements.each(function() {
      const $details = $(this);
      // Process any nested details first
      processDetailsRecursive($details);
    });
    
    // Then process current level details
    $detailsElements.each(function() {
      const $details = $(this);
      const $summary = $details.find("summary");
      const summaryContent = $summary.html() || $summary.text();
      
      // Extract style attributes from summary for heading styling
      const summaryStyle = $summary.attr("style") || "";
      let headingLevel = 3;
      let additionalClasses = [];
      
      // Parse font-size from style to determine heading level
      const fontSizeMatch = summaryStyle.match(/font-size:\s*([\d.]+)em/);
      if (fontSizeMatch) {
        const fontSize = parseFloat(fontSizeMatch[1]);
        if (fontSize >= 1.5) {
          headingLevel = 2;
        } else if (fontSize >= 1.2) {
          headingLevel = 3;
        } else {
          headingLevel = 4;
        }
      }
      
      // Check for italic styling
      if (summaryStyle.includes("font-style: italic") || summaryStyle.includes("italic")) {
        additionalClasses.push("italic-summary");
      }
      
      // Create heading tag
      const headingTag = `h${Math.min(6, headingLevel)}`;
      const headingClass = additionalClasses.length > 0 ? ` class="${additionalClasses.join(" ")}"` : "";
      const headingHtml = `<${headingTag}${headingClass}>${summaryContent}</${headingTag}>`;
      
      // Get the content inside details (excluding summary)
      const content = $details.html() || "";
      
      // Remove the summary element and its text content from the HTML
      $summary.remove();
      const contentWithoutSummary = $details.html() || "";
      
      // Clean up any remaining summary tags just in case
      const cleanedContent = contentWithoutSummary.replace(/<\/?summary[^>]*>/g, "");
      
      // Replace the details element with heading + content
      $details.replaceWith(headingHtml + cleanedContent);
    });
  }
  
  // Start processing from the root
  processDetailsRecursive($.root());

  return $.html();
}

/**
 * Fixes image paths to use base64 data URLs for PDF generation
 * This ensures images are embedded directly in the PDF
 */
async function fixImagePaths(html, lessonSlug) {
  const $ = cheerio.load(html, { xmlMode: false });
  const $images = $("img");
  
  if ($images.length === 0) {
    return html; // No images to process
  }
  
  // Process each image
  for (let i = 0; i < $images.length; i++) {
    const $img = $images.eq(i);
    const src = $img.attr("src");
    
    if (!src || !src.startsWith("/images/")) {
      continue; // Skip images that don't need processing
    }
    
    try {
      // Extract the path part after /images/
      const imagePathMatch = src.match(/\/images\/([^\/]+)\/([^"?#]+)/);
      if (!imagePathMatch) {
        continue;
      }
      
      const [, lessonDir, filename] = imagePathMatch;
      const actualLessonDir = LESSON_IMAGE_MAP[lessonDir] || lessonDir;
      const fullImagePath = path.join(process.cwd(), "public", "images", actualLessonDir, filename);
      
      // Check if file exists
      try {
        await fs.access(fullImagePath);
      } catch (error) {
        continue;
      }
      
      // Read the image file and convert to base64
      const imageBuffer = await fs.readFile(fullImagePath);
      const base64Data = imageBuffer.toString("base64");
      
      // Determine MIME type from file extension
      const extension = path.extname(filename).toLowerCase();
      let mimeType = "image/png";
      
      switch (extension) {
        case ".jpg":
        case ".jpeg":
          mimeType = "image/jpeg";
          break;
        case ".svg":
          mimeType = "image/svg+xml";
          break;
        case ".webp":
          mimeType = "image/webp";
          break;
        case ".gif":
          mimeType = "image/gif";
          break;
        case ".png":
          mimeType = "image/png";
          break;
        default:
          mimeType = "image/png";
      }
      
      // Create data URL
      const dataUrl = `data:${mimeType};base64,${base64Data}`;
      $img.attr("src", dataUrl);
      
    } catch (error) {
      // If there's an error, leave the original src attribute
    }
  }
  
  return $.html();
}

/**
 * Processes HTML to make it suitable for PDF generation
 */
async function processHTML(html, lessonSlug) {
  // First expand details sections
  html = expandDetails(html);
  
  // Then fix image paths
  html = await fixImagePaths(html, lessonSlug);
  
  // Remove any onclick handlers or JavaScript
  html = html.replace(/\s+onclick="[^"]*"/g, "");
  html = html.replace(/\s+style="[^"]*"/g, (match) => {
    // Keep only safe CSS properties
    const safeStyles = match.match(/style="([^"]*)"/);
    if (!safeStyles) return "";
    
    let styleContent = safeStyles[1];
    // Remove potentially problematic styles
    styleContent = styleContent.replace(/display:\s*none;/g, "");
    styleContent = styleContent.replace(/visibility:\s*hidden;/g, "");
    
    if (styleContent.trim()) {
      return ` style="${styleContent.trim()}"`;
    }
    return "";
  });
  
  return html;
}

/**
 * Generates combined HTML for a lesson section
 */
async function generateCombinedHTML(lessonFiles, sectionTitle, sectionSlug) {
  const cssPath = path.join(process.cwd(), "pdf", "styles.css");
  const css = await fs.readFile(cssPath, "utf-8");
  
  let htmlParts = [`
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${sectionTitle}</title>
  <style>
    ${css}
  </style>
</head>
<body>
  <h1>${sectionTitle}</h1>
`];

  // Group by subsections if available
  const processedLessons = new Set();
  
  // Add each lesson's content
  for (const lesson of lessonFiles) {
    if (processedLessons.has(lesson.path)) continue;
    processedLessons.add(lesson.path);
    
    const fileContent = await fs.readFile(lesson.path, "utf-8");
    
    // Parse frontmatter to get title
    const matterMatch = fileContent.match(/^---\s*\n([\s\S]*?)\n---/);
    const hasFrontmatter = matterMatch && matterMatch[1].trim();
    
    let markdownContent = fileContent;
    if (hasFrontmatter) {
      markdownContent = fileContent.replace(/^---\s*\n([\s\S]*?)\n---/, "");
    }
    
    // Convert markdown to HTML
    const rawHTML = marked.parse(markdownContent);
    
    // Process the HTML for PDF
    const processedHTML = await processHTML(rawHTML, sectionSlug);
    
    // Add section header with lesson title
    const lessonTitle = lesson.title || lesson.slug;
    htmlParts.push(`
  <div class="lesson-section" data-slug="${lesson.slug}">
    <h2 class="lesson-title">${lessonTitle}</h2>
    ${processedHTML}
  </div>
`);
  }

  htmlParts.push(`
</body>
</html>
`);
  
  return htmlParts.join("");
}

/**
 * Generates PDF from HTML using puppeteer
 */
async function generatePDF(browser, html, outputPath) {
  // Ensure output directory exists
  const outputDir = path.dirname(outputPath);
  await fs.mkdir(outputDir, { recursive: true });

  const page = await browser.newPage();
  
  // Set viewport for PDF generation
  await page.setViewport({
    width: 1200,
    height: 800,
    deviceScaleFactor: 2,
  });
  
  // Set HTML content
  await page.setContent(html, {
    waitUntil: "networkidle0",
  });
  
  // Wait for images to load
  await page.waitForFunction(() => {
    const images = document.querySelectorAll("img");
    return Array.from(images).every(img => img.complete);
  }, { timeout: 60000, polling: 500 });
  
  // Generate PDF
  await page.pdf({
    path: outputPath,
    format: "A4",
    margin: {
      top: "20mm",
      right: "20mm",
      bottom: "20mm",
      left: "20mm",
    },
    printBackground: true,
    preferCSSPageSize: true,
  });
  
  await page.close();
}

/**
 * Main function to generate PDFs for all lesson directories
 */
export async function generateLessonPDFs() {
  console.log("Starting PDF generation...");
  
  const sections = await getLessons();
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--disable-gpu", "--no-sandbox"],
  });
  
  try {
    for (const section of sections) {
      // Only process sections 02, 03, 04, 05 (skip introduction and others)
      const sectionNumber = section.order;
      if (!["02", "03", "04", "05"].includes(sectionNumber)) {
        console.log(`Skipping section ${section.title} (${section.slug}) - not in target range`);
        continue;
      }
      
      // Collect all markdown files in order
      const allMarkdownFiles = [];
      
      for (const subsection of section.subsections) {
        for (const lesson of subsection.lessons) {
          if (lesson.path.endsWith(".md")) {
            allMarkdownFiles.push(lesson);
          }
        }
      }
      
      if (allMarkdownFiles.length === 0) {
        console.log(`No markdown files found in section ${section.title}, skipping.`);
        continue;
      }
      
      console.log(`Processing section: ${section.title} (${allMarkdownFiles.length} files)`);
      
      // Generate combined HTML for this section
      const combinedHTML = await generateCombinedHTML(allMarkdownFiles, section.title, section.slug);
      
      // Generate PDF
      const pdfPath = path.join(process.cwd(), "out", "lessons", `${section.slug}.pdf`);
      console.log(`  Generating PDF: ${pdfPath}`);
      
      await generatePDF(browser, combinedHTML, pdfPath);
      
      console.log(`  ✓ Generated PDF: ${pdfPath}`);
    }
  } catch (error) {
    console.error("Error during PDF generation:", error);
    throw error;
  } finally {
    await browser.close();
  }
  
  console.log("PDF generation completed!");
}

// Run if called directly
generateLessonPDFs().catch(console.error);