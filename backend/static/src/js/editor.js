import { Editor } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';

window.setupEditor = function(container, initialMarkdown) {
  console.log('Setting up editor with:', {
    container,
    initialMarkdown
  });

  if (!container) {
    console.error('Editor container is null');
    return null;
  }

  try {
    const editor = new Editor({
      element: container,
      extensions: [
        StarterKit,
        Placeholder.configure({
          placeholder: 'Start writing...'
        })
      ],
      content: initialMarkdown,
      editable: true,
      autofocus: true,
      onCreate: ({ editor }) => {
        console.log('Editor created successfully');
      },
      onUpdate: ({ editor }) => {
        console.log('Content updated:', editor.getHTML());
      }
    });

    return {
      getMarkdown: () => editor.getHTML(),
      setMarkdown: (content) => editor.commands.setContent(content),
      destroy: () => editor.destroy()
    };
  } catch (error) {
    console.error('Error creating editor:', error);
    return null;
  }
}; 