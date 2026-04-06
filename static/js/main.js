// SmartStudy AI - main.js
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".flash").forEach(el => {
    setTimeout(() => el && el.remove(), 4000);
  });
  const toggle = document.getElementById("navToggle");
  const links  = document.getElementById("navLinks");
  if (toggle && links) toggle.addEventListener("click", () => links.classList.toggle("open"));
});
