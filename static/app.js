document.addEventListener("DOMContentLoaded", () => {
  const links = document.querySelectorAll("nav a");
  const sections = document.querySelectorAll("main section");

  links.forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const targetId = link.getAttribute("href").replace("#", "");

      // Hide all sections
      sections.forEach((section) => section.classList.add("hidden"));

      // Show the target section
      document.getElementById(targetId).classList.remove("hidden");
    });
  });

  // Show the first section by default
  sections.forEach((section, index) => {
    if (index === 0) {
      section.classList.remove("hidden");
    } else {
      section.classList.add("hidden");
    }
  });
});
