import java.awt.*;
import java.awt.event.*;

public class MouseListenerExample extends Frame implements MouseListener {

    // Constructor
    MouseListenerExample() {
        setTitle("MouseListener Example");
        setSize(400, 300);
        
        // Create a button and panel
        Button button = new Button("Click Me");
        Panel panel = new Panel();
        
        // Set layout and add button to panel
        panel.setLayout(new FlowLayout());
        panel.add(button);
        
        // Add the panel to the frame
        add(panel);
        
        // Add MouseListener to the button
        button.addMouseListener(this);

        // Add MouseListener to the panel
        panel.addMouseListener(this);

        setVisible(true);
        addWindowListener(new WindowAdapter(){
        
            public void windowClosing(WindowEvent we)
            {
                dispose();
            }
        });
    }

    // Handle mouse click events
    public void mouseClicked(MouseEvent e) {
        System.out.println("Mouse Clicked on: " + e.getComponent());
    }

    // Handle mouse press events (currently not used in this example)
    public void mousePressed(MouseEvent e) { }

    // Handle mouse release events (currently not used in this example)
    public void mouseReleased(MouseEvent e) { }

    // Handle mouse entering the component
    public void mouseEntered(MouseEvent e) {
        System.out.println("Mouse Entered on: " + e.getComponent());
    }

    // Handle mouse exiting the component
    public void mouseExited(MouseEvent e) {
        System.out.println("Mouse Exited from: " + e.getComponent());
    }

    // Main method to run the example
    public static void main(String[] args) {
        new MouseListenerExample();
    }
}
